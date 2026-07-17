"""Fair subject-level metrics for multi-condition communication benchmarks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EDGE_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")
FAMILY_KEYS = ("sender", "receiver")
MOLECULAR_LR_EQUIVALENCE_COLUMN = "molecular_lr_equivalence_id"
METHOD_IDENTITY_KEYS = (
    "dataset",
    "method",
    "method_version",
    "analysis_track",
    "resource",
    "resource_version",
    "resource_mode",
    "score_semantics",
    "universe_id",
)
PERFORMANCE_IDENTITY_KEYS = (
    "dataset",
    "method",
    "method_version",
    "analysis_track",
    "resource",
    "resource_version",
    "resource_mode",
)
RESOURCE_MODES = frozenset({"H-common", "H-covered", "native"})
SCORE_STATUSES = frozenset(
    {
        "observed",
        "not_predicted",
        "missing",
        "resource_unavailable",
        "cell_type_missing",
        "not_estimable",
        "filtered",
        "failed",
        "not_supported",
    }
)
COMPARISON_ELIGIBLE_STATUSES = frozenset({"observed", "not_predicted"})
EFFECT_ESTIMABLE_STATUSES = frozenset({"observed", "exploratory", "descriptive"})
SYNTHETIC_TRUTH_SCOPES = frozenset(
    {"simulation", "synthetic", "perturbation_with_known_truth"}
)
REQUIRED_SCORE_COLUMNS = {
    *METHOD_IDENTITY_KEYS,
    "sample_id",
    "subject_id",
    "context",
    "contrast",
    *EDGE_KEYS,
    "score",
    "score_direction",
    "status",
    "universe_member",
    "universe_size",
}


def _one_value(values: pd.Series, *, label: str) -> object:
    unique = values.drop_duplicates()
    if len(unique) != 1:
        raise ValueError(f"{label} must have exactly one value per score identity")
    return unique.iloc[0]


def _validate_fixed_universe(table: pd.DataFrame) -> None:
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    for _, method_group in table.groupby(grouping, sort=False, observed=True):
        if not method_group["universe_member"].all():
            raise ValueError(
                "metric score tables may contain only frozen-universe members"
            )
        declared_sizes = method_group["universe_size"].drop_duplicates()
        if len(declared_sizes) != 1:
            raise ValueError("universe_size must be constant per score identity")
        declared_size = int(declared_sizes.iloc[0])
        actual_size = len(method_group.loc[:, EDGE_KEYS].drop_duplicates())
        if declared_size != actual_size:
            raise ValueError(
                "universe_size does not match the materialized frozen edge universe"
            )
        sample_sizes = method_group.groupby(
            "sample_id", sort=False, observed=True
        ).size()
        if not sample_sizes.eq(declared_size).all():
            raise ValueError(
                "every sample must materialize the same fixed edge universe; "
                "represent absent results with an explicit status row"
            )

        unavailable = method_group["status"].eq("resource_unavailable")
        unavailable_state = unavailable.groupby(
            [method_group[key] for key in EDGE_KEYS],
            sort=False,
            observed=True,
        ).agg(["any", "all"])
        if (unavailable_state["any"] & ~unavailable_state["all"]).any():
            raise ValueError(
                "resource_unavailable is a frozen method-resource edge state "
                "and cannot vary by sample"
            )


def _validate_molecular_lr_equivalence_mapping(table: pd.DataFrame) -> None:
    if MOLECULAR_LR_EQUIVALENCE_COLUMN not in table.columns:
        return

    equivalence_ids = table[MOLECULAR_LR_EQUIVALENCE_COLUMN]
    canonical = equivalence_ids.map(
        lambda value: (
            isinstance(value, str) and bool(value) and value == value.strip()
        )
    )
    if not canonical.all():
        raise ValueError(
            "molecular_lr_equivalence_id must contain canonical non-empty strings"
        )

    source_edge_key = [*PERFORMANCE_IDENTITY_KEYS, *EDGE_KEYS]
    mappings = table.loc[
        :, [*source_edge_key, MOLECULAR_LR_EQUIVALENCE_COLUMN]
    ].drop_duplicates()
    if mappings.duplicated(source_edge_key, keep=False).any():
        raise ValueError(
            "each method-resource source edge must map to exactly one "
            "molecular_lr_equivalence_id across samples and contrasts"
        )


def _add_comparison_values(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    result["comparison_eligible"] = result["status"].isin(COMPARISON_ELIGIBLE_STATUSES)
    result["_resource_eligible"] = ~result["status"].eq("resource_unavailable")
    result["_observed"] = result["status"].eq("observed")
    direction = result["score_direction"].map({"higher": 1.0, "lower": -1.0})
    result["_oriented_score"] = result["score"] * direction
    grouping = [*METHOD_IDENTITY_KEYS, "contrast", "sample_id"]
    grouped = result.groupby(grouping, sort=False, observed=True)
    eligible_size = grouped["comparison_eligible"].transform("sum").astype(int)
    observed_size = grouped["_observed"].transform("sum").astype(int)
    result["eligible_universe_size"] = (
        grouped["_resource_eligible"].transform("sum").astype(int)
    )
    result["comparison_eligible_size"] = eligible_size
    result["comparison_rank"] = grouped["_oriented_score"].rank(
        ascending=False, method="average"
    )
    not_predicted = result["status"].eq("not_predicted")
    result.loc[not_predicted, "comparison_rank"] = (
        observed_size.loc[not_predicted] + 1 + eligible_size.loc[not_predicted]
    ) / 2
    result["comparison_strength"] = np.nan
    observed = result["_observed"]
    # Observed values occupy (0, 1]; zero is reserved for not-predicted.
    result.loc[observed, "comparison_strength"] = 1.0 - (
        result.loc[observed, "comparison_rank"] - 1.0
    ) / eligible_size.loc[observed]
    result.loc[not_predicted, "comparison_strength"] = 0.0
    result = result.drop(
        columns=["_resource_eligible", "_observed", "_oriented_score"]
    )
    derived = [
        "comparison_eligible",
        "comparison_rank",
        "comparison_strength",
        "eligible_universe_size",
        "comparison_eligible_size",
    ]
    original = [column for column in result.columns if column not in derived]
    return result.loc[:, original + derived]


def validate_score_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate, orient, and rank a sample-level LR score table.

    ``score`` always remains the method's native value. ``comparison_rank`` is
    one-best with average ranks for ties, while ``comparison_strength`` is a
    direction-normalized rank strength in ``[0, 1]``. Only ``observed`` and
    ``not_predicted`` rows enter comparisons; the latter are tied below every
    observed result and have comparison strength zero.
    """

    missing = REQUIRED_SCORE_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"score table is missing columns: {sorted(missing)}")
    result = table.copy(deep=True)
    metadata = list(REQUIRED_SCORE_COLUMNS - {"score"})
    if result[metadata].isna().any().any():
        raise ValueError(
            "score table metadata and edge identifiers must not be missing"
        )
    if not set(result["score_direction"].astype(str)).issubset({"higher", "lower"}):
        raise ValueError("score_direction must be 'higher' or 'lower'")
    if set(result["analysis_track"].astype(str)) != {"lr_stlr"}:
        raise ValueError(
            "multi-condition LR metrics require analysis_track='lr_stlr'; "
            "ligand-target outputs require Track B metrics"
        )
    invalid_modes = set(result["resource_mode"].astype(str)).difference(RESOURCE_MODES)
    if invalid_modes:
        raise ValueError(
            f"resource_mode contains invalid arms: {sorted(invalid_modes)}"
        )
    if not pd.api.types.is_bool_dtype(result["universe_member"]):
        raise ValueError("universe_member must be boolean")
    universe_size = pd.to_numeric(result["universe_size"], errors="coerce")
    if (
        universe_size.isna().any()
        or (universe_size < 1).any()
        or (universe_size % 1 != 0).any()
    ):
        raise ValueError("universe_size must contain positive integers")
    result["universe_size"] = universe_size.astype(int)
    invalid_statuses = set(result["status"].astype(str)).difference(SCORE_STATUSES)
    if invalid_statuses:
        raise ValueError(
            f"score table contains invalid statuses: {sorted(invalid_statuses)}"
        )

    raw_score = pd.to_numeric(result["score"], errors="coerce")
    supplied = result["score"].notna()
    if ((supplied & raw_score.isna()) | np.isinf(raw_score.fillna(0.0))).any():
        raise ValueError("observed score values must be finite numeric values")
    observed = result["status"].eq("observed")
    if raw_score[observed].isna().any():
        raise ValueError("status observed requires a finite native score")
    if raw_score[~observed].notna().any():
        raise ValueError("non-observed statuses require a missing native score")
    result["score"] = raw_score

    key = [*METHOD_IDENTITY_KEYS, "contrast", "sample_id", *EDGE_KEYS]
    if result.duplicated(key).any():
        raise ValueError(
            "score table contains duplicate method-sample-contrast-LR rows"
        )
    sample_map = result[
        ["dataset", "sample_id", "subject_id", "context"]
    ].drop_duplicates()
    if sample_map.duplicated(["dataset", "sample_id"]).any():
        raise ValueError(
            "each dataset/sample_id must map to exactly one subject and context"
        )
    direction_groups = [*METHOD_IDENTITY_KEYS, "contrast"]
    if (
        result.groupby(direction_groups, observed=True)["score_direction"].nunique() > 1
    ).any():
        raise ValueError("score_direction must be constant within each score identity")
    _validate_molecular_lr_equivalence_mapping(result)
    _validate_fixed_universe(result)
    ordered = result.sort_values(key, kind="stable", ignore_index=True)
    return _add_comparison_values(ordered)


def _prepared_score_table(
    table: pd.DataFrame, *, validated: bool
) -> pd.DataFrame:
    if not validated:
        return validate_score_table(table)
    prepared_columns = {
        "comparison_eligible",
        "comparison_rank",
        "comparison_strength",
        "eligible_universe_size",
        "comparison_eligible_size",
    }
    missing = prepared_columns.difference(table.columns)
    if missing:
        raise ValueError(
            "validated score table is missing prepared columns: "
            f"{sorted(missing)}"
        )
    return table


def external_long_to_score_table(
    table: pd.DataFrame,
    *,
    context_key: str,
    contrast: str,
    dataset: str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Map the external adapter long-table contract to the metric contract.

    This conversion never guesses why a sparse result is absent. Before using
    the default ``validate=True``, callers must outer-join adapter output to the
    frozen sample-by-edge universe and classify every absent row as one of the
    canonical statuses. ``validate=False`` is provided only to obtain the mapped
    observed rows before that explicit materialization step.
    """

    required = {
        "method_id",
        "method_version",
        "analysis_track",
        "resource_id",
        "resource_version",
        "resource_mode",
        "universe_id",
        "universe_member",
        "universe_size",
        "sample_id",
        "subject_id",
        "context_json",
        *EDGE_KEYS,
        "score",
        "score_name",
        "score_direction",
        "status",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"external long table is missing columns: {sorted(missing)}")
    if set(table["analysis_track"].astype(str)) != {"lr_stlr"}:
        raise ValueError(
            "external LR score conversion accepts only analysis_track='lr_stlr'; "
            "route ligand-target output to Track B"
        )
    if "dataset_id" in table:
        external_datasets = table["dataset_id"].astype(str)
        if dataset is not None and set(external_datasets) != {dataset}:
            raise ValueError("dataset disagrees with external dataset_id")
    elif dataset is not None:
        external_datasets = pd.Series(dataset, index=table.index, dtype=str)
    else:
        raise ValueError("external long table requires dataset_id or dataset")
    context: list[str] = []
    for value in table["context_json"]:
        parsed = json.loads(str(value))
        if not isinstance(parsed, dict) or context_key not in parsed:
            raise ValueError(
                f"context_json must encode an object containing {context_key!r}"
            )
        context.append(str(parsed[context_key]))
    status_map = {
        "ok": "observed",
        "missing": "missing",
        "not_returned": "not_predicted",
        "unsupported_resource": "not_supported",
        "insufficient_cells": "cell_type_missing",
        "method_failed": "failed",
    }
    result = pd.DataFrame(
        {
            "dataset": external_datasets,
            "method": table["method_id"].astype(str),
            "method_version": table["method_version"].astype(str),
            "analysis_track": table["analysis_track"].astype(str),
            "resource": table["resource_id"].astype(str),
            "resource_version": table["resource_version"].astype(str),
            "resource_mode": table["resource_mode"].astype(str),
            "sample_id": table["sample_id"].astype(str),
            "subject_id": table["subject_id"].astype(str),
            "context": context,
            "contrast": contrast,
            "sender": table["sender"].astype(str),
            "receiver": table["receiver"].astype(str),
            "interaction_id": table["interaction_id"].astype(str),
            "ligand": table["ligand"].astype(str),
            "receptor": table["receptor"].astype(str),
            "score": table["score"],
            "score_direction": table["score_direction"].astype(str),
            "score_semantics": table["score_name"].astype(str),
            "status": table["status"].astype(str).replace(status_map),
            "universe_id": table["universe_id"].astype(str),
            "universe_member": table["universe_member"],
            "universe_size": table["universe_size"],
        }
    )
    if MOLECULAR_LR_EQUIVALENCE_COLUMN in table.columns:
        result[MOLECULAR_LR_EQUIVALENCE_COLUMN] = table[
            MOLECULAR_LR_EQUIVALENCE_COLUMN
        ]
    return validate_score_table(result) if validate else result


def score_coverage_summary(
    table: pd.DataFrame, *, validated: bool = False
) -> pd.DataFrame:
    """Report resource and result coverage without converting absence to zero."""

    scores = _prepared_score_table(table, validated=validated)
    rows: list[dict[str, object]] = []
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    for keys, group in scores.groupby(grouping, sort=False, observed=True):
        frozen_size = int(
            str(_one_value(group["universe_size"], label="universe_size"))
        )
        edge_status = group.groupby(list(EDGE_KEYS), sort=False, observed=True)[
            "status"
        ]
        unavailable_edges = int(
            sum(edge.eq("resource_unavailable").all() for _, edge in edge_status)
        )
        resource_covered_edges = frozen_size - unavailable_edges
        resource_eligible = ~group["status"].eq("resource_unavailable")
        comparable = group["comparison_eligible"]
        sample_coverage: list[float] = []
        for _, sample in group.groupby("sample_id", sort=False, observed=True):
            denominator = int((~sample["status"].eq("resource_unavailable")).sum())
            sample_coverage.append(
                float(sample["comparison_eligible"].sum() / denominator)
                if denominator
                else math.nan
            )
        finite_sample_coverage = np.asarray(sample_coverage, dtype=float)
        finite_sample_coverage = finite_sample_coverage[
            np.isfinite(finite_sample_coverage)
        ]
        identity = dict(zip(grouping, keys, strict=True))
        identity.update(
            {
                "n_samples": int(group["sample_id"].nunique()),
                "n_subjects": int(group["subject_id"].nunique()),
                "frozen_universe_edges": frozen_size,
                "resource_covered_edges": resource_covered_edges,
                "resource_coverage_fraction": resource_covered_edges / frozen_size,
                "eligible_universe_rows": int(resource_eligible.sum()),
                "comparison_eligible_rows": int(comparable.sum()),
                "comparison_coverage_fraction": (
                    float(comparable.sum() / resource_eligible.sum())
                    if resource_eligible.any()
                    else math.nan
                ),
                "observed_rows": int(group["status"].eq("observed").sum()),
                "not_predicted_rows": int(group["status"].eq("not_predicted").sum()),
                "missing_rows": int(group["status"].eq("missing").sum()),
                "cell_type_missing_rows": int(
                    group["status"].eq("cell_type_missing").sum()
                ),
                "not_estimable_rows": int(group["status"].eq("not_estimable").sum()),
                "failed_rows": int(group["status"].eq("failed").sum()),
                "not_supported_rows": int(group["status"].eq("not_supported").sum()),
                "median_sample_comparison_coverage": (
                    float(np.median(finite_sample_coverage))
                    if len(finite_sample_coverage)
                    else math.nan
                ),
                "minimum_sample_comparison_coverage": (
                    float(np.min(finite_sample_coverage))
                    if len(finite_sample_coverage)
                    else math.nan
                ),
                "status": "observed" if comparable.any() else "not_estimable",
                "reason_code": (
                    None if comparable.any() else "no_comparison_eligible_scores"
                ),
            }
        )
        rows.append(identity)
    return pd.DataFrame(rows)


def _top_keys(table: pd.DataFrame, *, k: int) -> set[tuple[object, ...]]:
    ranked = table.loc[table["comparison_eligible"] & table["comparison_rank"].notna()]
    if ranked.empty:
        return set()
    ordered_ranks = ranked["comparison_rank"].sort_values(kind="stable")
    cutoff = float(ordered_ranks.iloc[min(k, len(ordered_ranks)) - 1])
    selected = ranked.loc[ranked["comparison_rank"] <= cutoff]
    return set(selected.loc[:, EDGE_KEYS].itertuples(index=False, name=None))


def within_context_reproducibility(
    table: pd.DataFrame, *, top_k: int = 100, validated: bool = False
) -> pd.DataFrame:
    """Summarize pairwise sample rank/Jaccard reproducibility within contexts."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    scores = _prepared_score_table(table, validated=validated)
    rows: list[dict[str, object]] = []
    grouping = [*METHOD_IDENTITY_KEYS, "contrast", "context"]
    for keys, group in scores.groupby(grouping, sort=False, observed=True):
        samples = tuple(sorted(group["sample_id"].unique()))
        rho_values: list[float] = []
        jaccard_values: list[float] = []
        shared_values: list[int] = []
        for left_id, right_id in itertools.combinations(samples, 2):
            left = group.loc[group["sample_id"] == left_id]
            right = group.loc[group["sample_id"] == right_id]
            merged = left[
                [*EDGE_KEYS, "comparison_strength", "comparison_eligible"]
            ].merge(
                right[[*EDGE_KEYS, "comparison_strength", "comparison_eligible"]],
                on=list(EDGE_KEYS),
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            comparable = merged.loc[
                merged["comparison_eligible_left"] & merged["comparison_eligible_right"]
            ]
            shared_values.append(len(comparable))
            if (
                len(comparable) >= 3
                and comparable["comparison_strength_left"].nunique() >= 2
                and comparable["comparison_strength_right"].nunique() >= 2
            ):
                rho = spearmanr(
                    comparable["comparison_strength_left"],
                    comparable["comparison_strength_right"],
                ).statistic
                if np.isfinite(rho):
                    rho_values.append(float(rho))
            left_top = _top_keys(left, k=top_k)
            right_top = _top_keys(right, k=top_k)
            union = left_top | right_top
            if union:
                jaccard_values.append(len(left_top & right_top) / len(union))
        identity = dict(zip(grouping, keys, strict=True))
        enough_samples = len(samples) >= 2
        any_metric = bool(rho_values or jaccard_values)
        estimable = enough_samples and any_metric
        identity.update(
            {
                "n_samples": len(samples),
                "n_sample_pairs": math.comb(len(samples), 2),
                "median_shared_edges": (
                    float(np.median(shared_values)) if shared_values else np.nan
                ),
                "median_spearman": (
                    float(np.median(rho_values)) if rho_values else np.nan
                ),
                "median_top_k_jaccard": (
                    float(np.median(jaccard_values)) if jaccard_values else np.nan
                ),
                "top_k": top_k,
                "status": "observed" if estimable else "not_estimable",
                "reason_code": (
                    None
                    if estimable
                    else "fewer_than_two_samples"
                    if not enough_samples
                    else "no_pairwise_comparable_scores"
                ),
            }
        )
        rows.append(identity)
    return pd.DataFrame(rows)


def _select_contrast(scores: pd.DataFrame, contrast: str | None) -> pd.DataFrame:
    contrasts = tuple(scores["contrast"].drop_duplicates())
    if contrast is None:
        if len(contrasts) != 1:
            raise ValueError(
                "contrast must be provided when the table has multiple contrasts"
            )
        return scores
    selected = scores.loc[scores["contrast"].eq(contrast)]
    if selected.empty:
        raise ValueError(f"contrast {contrast!r} is absent from the score table")
    return selected


def _reason_for_unestimable(group: pd.DataFrame, n_pairs: int) -> str:
    if group["status"].eq("resource_unavailable").all():
        return "resource_unavailable"
    if group["status"].eq("not_supported").all():
        return "method_or_resource_not_supported"
    if group["status"].eq("failed").any():
        return "method_run_failed"
    if group["status"].eq("not_estimable").any() and n_pairs == 0:
        return "input_not_estimable"
    if n_pairs == 0 and group["status"].eq("cell_type_missing").any():
        return "cell_type_missing"
    if n_pairs == 0 and group["status"].eq("missing").any():
        return "missing_score"
    return "insufficient_paired_subjects"


def _normalize_subject_context(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    result["subject_id"] = result["subject_id"].astype(str)
    result["context"] = result["context"].astype(str)
    return result


def _validate_unpaired_contexts(
    group: pd.DataFrame,
    *,
    reference: str,
    target: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available_contexts = set(group["context"].astype(str))
    missing_contexts = {reference, target}.difference(available_contexts)
    if missing_contexts:
        raise ValueError(f"score table is missing contexts: {sorted(missing_contexts)}")
    support = group.loc[
        group["context"].isin((reference, target)), ["subject_id", "context"]
    ].drop_duplicates()
    reference_subjects = tuple(
        sorted(
            support.loc[support["context"].eq(reference), "subject_id"]
            .astype(str)
            .unique()
        )
    )
    target_subjects = tuple(
        sorted(
            support.loc[support["context"].eq(target), "subject_id"]
            .astype(str)
            .unique()
        )
    )
    overlap = set(reference_subjects) & set(target_subjects)
    if overlap:
        raise ValueError(
            "unpaired comparisons require disjoint subject groups; "
            f"subjects mapped to both contexts: {sorted(overlap)}"
        )
    return reference_subjects, target_subjects


def _unpaired_support_reason(
    group: pd.DataFrame,
    *,
    n_reference_subjects: int,
    n_target_subjects: int,
    min_subjects: int,
) -> str:
    if group["status"].eq("resource_unavailable").all():
        return "resource_unavailable"
    if group["status"].eq("not_supported").all():
        return "method_or_resource_not_supported"
    if group["status"].eq("failed").any():
        return "method_run_failed"
    reference_insufficient = n_reference_subjects < min_subjects
    target_insufficient = n_target_subjects < min_subjects
    if reference_insufficient and target_insufficient:
        return "insufficient_subjects_both_contexts"
    if reference_insufficient:
        return "insufficient_reference_subjects"
    if target_insufficient:
        return "insufficient_target_subjects"
    return "input_not_estimable"


def _subject_edge_strengths(group: pd.DataFrame) -> pd.Series:
    usable = group.loc[group["comparison_eligible"]]
    if usable.empty:
        index = pd.MultiIndex.from_arrays(
            [[] for _ in range(2 + len(EDGE_KEYS))],
            names=["subject_id", "context", *EDGE_KEYS],
        )
        return pd.Series(index=index, dtype=float, name="comparison_strength")
    return usable.groupby(
        ["subject_id", "context", *EDGE_KEYS],
        observed=True,
        sort=False,
    )["comparison_strength"].mean()


def _frozen_eligible_universe(group: pd.DataFrame) -> pd.DataFrame:
    edges = group.loc[:, list(EDGE_KEYS)].drop_duplicates(ignore_index=True)
    edge_index = pd.MultiIndex.from_frame(edges)
    unavailable = group["status"].eq("resource_unavailable")
    all_unavailable = unavailable.groupby(
        [group[key] for key in EDGE_KEYS],
        sort=False,
        observed=True,
    ).all()
    eligible = ~all_unavailable.reindex(edge_index).to_numpy(dtype=bool)
    return edges.loc[eligible].reset_index(drop=True)


def _edge_group_base(
    table: pd.DataFrame, grouping: Sequence[str]
) -> tuple[pd.DataFrame, pd.MultiIndex]:
    base = table.loc[:, list(grouping)].drop_duplicates(ignore_index=True)
    return base, pd.MultiIndex.from_frame(base)


def _edge_status_summary(
    table: pd.DataFrame, grouping: Sequence[str]
) -> pd.DataFrame:
    status = table["status"]
    flags = pd.DataFrame(
        {
            "n_not_predicted_rows": status.eq("not_predicted").astype(np.int8),
            "n_missing_rows": status.eq("missing").astype(np.int8),
            "n_resource_unavailable_rows": status.eq(
                "resource_unavailable"
            ).astype(np.int8),
            "n_cell_type_missing_rows": status.eq("cell_type_missing").astype(
                np.int8
            ),
            "_all_resource_unavailable": status.eq("resource_unavailable"),
            "_all_not_supported": status.eq("not_supported"),
            "_any_failed": status.eq("failed"),
            "_any_not_estimable": status.eq("not_estimable"),
            "_any_cell_type_missing": status.eq("cell_type_missing"),
            "_any_missing": status.eq("missing"),
        },
        index=table.index,
    )
    grouped = flags.groupby(
        [table[key] for key in grouping],
        sort=False,
        observed=True,
    )
    return grouped.agg(
        {
            "n_not_predicted_rows": "sum",
            "n_missing_rows": "sum",
            "n_resource_unavailable_rows": "sum",
            "n_cell_type_missing_rows": "sum",
            "_all_resource_unavailable": "all",
            "_all_not_supported": "all",
            "_any_failed": "any",
            "_any_not_estimable": "any",
            "_any_cell_type_missing": "any",
            "_any_missing": "any",
        }
    )


def _rowwise_distribution(
    values: np.ndarray,
    *,
    order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = ~np.isnan(values)
    counts = valid.sum(axis=1).astype(int)
    means: np.ndarray = np.full(len(values), np.nan, dtype=float)
    medians: np.ndarray = np.full(len(values), np.nan, dtype=float)
    scales: np.ndarray = np.full(len(values), np.nan, dtype=float)
    variances: np.ndarray = np.full(len(values), np.nan, dtype=float)
    for row in range(len(values)):
        columns = np.flatnonzero(valid[row])
        n_values = len(columns)
        if n_values == 0:
            continue
        if order is not None:
            positions = order[row, columns]
            columns = columns[np.argsort(positions, kind="stable")]
        compact = values[row, columns]
        means[row] = compact.mean()
        medians[row] = np.median(compact)
        if n_values >= 2:
            scales[row] = compact.std(ddof=1)
            variances[row] = compact.var(ddof=1)
    return counts, means, medians, scales, variances


def _context_subject_values(
    subject_scores: pd.Series,
    subject_order: pd.Series,
    *,
    edge_index: pd.MultiIndex,
    identity_keys: Sequence[str],
    context: str,
    subjects: tuple[str, ...],
    n_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    if subject_scores.empty:
        shape = (n_edges, len(subjects))
        return (
            np.full(shape, np.nan, dtype=float),
            np.full(shape, np.nan, dtype=float),
        )
    contexts = subject_scores.index.get_level_values("context")
    if context not in contexts:
        shape = (n_edges, len(subjects))
        return (
            np.full(shape, np.nan, dtype=float),
            np.full(shape, np.nan, dtype=float),
        )
    values = subject_scores.xs(context, level="context").unstack("subject_id")
    positions = subject_order.xs(context, level="context").unstack("subject_id")
    edge_only_index = edge_index.droplevel(list(identity_keys))
    return (
        values.reindex(
            index=edge_only_index,
            columns=subjects,
        ).to_numpy(dtype=float),
        positions.reindex(
            index=edge_only_index,
            columns=subjects,
        ).to_numpy(dtype=float),
    )


def _paired_reason_codes(
    status_summary: pd.DataFrame,
    n_pairs: np.ndarray,
    estimable: np.ndarray,
) -> np.ndarray:
    reason: np.ndarray = np.full(len(status_summary), None, dtype=object)
    unresolved = ~estimable
    rules = (
        (
            status_summary["_all_resource_unavailable"].to_numpy(dtype=bool),
            "resource_unavailable",
        ),
        (
            status_summary["_all_not_supported"].to_numpy(dtype=bool),
            "method_or_resource_not_supported",
        ),
        (status_summary["_any_failed"].to_numpy(dtype=bool), "method_run_failed"),
        (
            status_summary["_any_not_estimable"].to_numpy(dtype=bool)
            & (n_pairs == 0),
            "input_not_estimable",
        ),
        (
            status_summary["_any_cell_type_missing"].to_numpy(dtype=bool)
            & (n_pairs == 0),
            "cell_type_missing",
        ),
        (
            status_summary["_any_missing"].to_numpy(dtype=bool) & (n_pairs == 0),
            "missing_score",
        ),
    )
    for mask, code in rules:
        selected = unresolved & mask
        reason[selected] = code
        unresolved &= ~selected
    reason[unresolved] = "insufficient_paired_subjects"
    return reason


def _unpaired_reason_codes(
    status_summary: pd.DataFrame,
    *,
    n_reference: np.ndarray,
    n_target: np.ndarray,
    min_subjects: int,
    estimable: np.ndarray,
) -> np.ndarray:
    reason: np.ndarray = np.full(len(status_summary), None, dtype=object)
    unresolved = ~estimable
    rules = (
        (
            status_summary["_all_resource_unavailable"].to_numpy(dtype=bool),
            "resource_unavailable",
        ),
        (
            status_summary["_all_not_supported"].to_numpy(dtype=bool),
            "method_or_resource_not_supported",
        ),
        (status_summary["_any_failed"].to_numpy(dtype=bool), "method_run_failed"),
        (
            (n_reference < min_subjects) & (n_target < min_subjects),
            "insufficient_subjects_both_contexts",
        ),
        (n_reference < min_subjects, "insufficient_reference_subjects"),
        (n_target < min_subjects, "insufficient_target_subjects"),
    )
    for mask, code in rules:
        selected = unresolved & mask
        reason[selected] = code
        unresolved &= ~selected
    reason[unresolved] = "input_not_estimable"
    return reason


def _paired_identity_effects(
    subset: pd.DataFrame,
    *,
    grouping: Sequence[str],
    reference: str,
    target: str,
    min_pairs: int,
) -> pd.DataFrame:
    base, edge_index = _edge_group_base(subset, grouping)
    usable = subset.loc[subset["comparison_eligible"]]
    subject_grouping = [*grouping, "subject_id", "context"]
    row_positions = pd.Series(
        np.arange(len(subset), dtype=int),
        index=subset.index,
    )
    subject_scores = usable.groupby(
        [usable[key] for key in subject_grouping],
        observed=True,
        sort=False,
    )["comparison_strength"].mean()
    subject_first = row_positions.loc[usable.index].groupby(
        [usable[key] for key in subject_grouping],
        observed=True,
        sort=False,
    ).min()
    if subject_scores.empty:
        differences = pd.DataFrame(index=edge_index, dtype=float)
        difference_order = pd.DataFrame(index=edge_index, dtype=float)
    else:
        subject_context = subject_scores.unstack("context")
        context_values = subject_context.reindex(columns=[reference, target])
        paired = context_values[reference].notna() & context_values[target].notna()
        difference = (
            context_values.loc[paired, target]
            - context_values.loc[paired, reference]
        )
        subject_order = tuple(subset["subject_id"].drop_duplicates())
        differences = difference.unstack("subject_id").reindex(
            index=edge_index,
            columns=subject_order,
        )
        first_context = subject_first.unstack("context").reindex(
            columns=[reference, target]
        )
        first_subject = first_context.loc[paired].min(axis=1)
        difference_order = first_subject.unstack("subject_id").reindex(
            index=edge_index,
            columns=subject_order,
        )
    difference_values = differences.to_numpy(dtype=float)
    n_pairs, effects, medians, scales, _ = _rowwise_distribution(
        difference_values,
        order=difference_order.to_numpy(dtype=float),
    )
    standard_errors: np.ndarray = np.full(len(base), np.nan, dtype=float)
    replicated = n_pairs >= 2
    for row in np.flatnonzero(replicated):
        standard_errors[row] = float(scales[row] / math.sqrt(int(n_pairs[row])))
    standardized: np.ndarray = np.full(len(base), np.nan, dtype=float)
    scalable = np.isfinite(scales) & (scales > 0)
    for row in np.flatnonzero(scalable):
        standardized[row] = effects[row] / scales[row]

    valid = ~np.isnan(difference_values)
    nonzero = valid & (difference_values != 0)
    direction_comparable = nonzero.sum(axis=1).astype(int)
    positive = (difference_values > 0).sum(axis=1)
    negative = (difference_values < 0).sum(axis=1)
    direction_consistency: np.ndarray = np.full(len(base), np.nan, dtype=float)
    positive_effect = (effects > 0) & (direction_comparable > 0)
    negative_effect = (effects < 0) & (direction_comparable > 0)
    direction_consistency[positive_effect] = (
        positive[positive_effect] / direction_comparable[positive_effect]
    )
    direction_consistency[negative_effect] = (
        negative[negative_effect] / direction_comparable[negative_effect]
    )

    status_summary = _edge_status_summary(subset, grouping).reindex(edge_index)
    estimable = n_pairs >= min_pairs
    result = base.copy()
    result["reference"] = reference
    result["target"] = target
    result["effect"] = effects
    result["effect_semantics"] = "target_minus_reference_comparison_strength"
    result["diagnostic_standard_error"] = standard_errors
    result["standardized_effect"] = standardized
    result["median_effect"] = medians
    result["direction_consistency"] = direction_consistency
    result["direction_comparable_pairs"] = direction_comparable
    result["n_pairs"] = n_pairs
    for column in (
        "n_not_predicted_rows",
        "n_missing_rows",
        "n_resource_unavailable_rows",
        "n_cell_type_missing_rows",
    ):
        result[column] = status_summary[column].to_numpy(dtype=int)
    result["status"] = np.where(estimable, "exploratory", "not_estimable")
    result["reason_code"] = _paired_reason_codes(
        status_summary, n_pairs, estimable
    )
    return result


def paired_edge_effects(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    min_pairs: int = 3,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Compute paired target-reference comparison-strength effects by subject."""

    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 2:
        raise ValueError("min_pairs must be an integer >= 2")
    scores = _normalize_subject_context(
        _select_contrast(
            _prepared_score_table(table, validated=validated), contrast
        )
    )
    if reference == target:
        raise ValueError("reference and target contexts must differ")
    available_contexts = set(scores["context"])
    missing_contexts = {reference, target}.difference(available_contexts)
    if missing_contexts:
        raise ValueError(f"score table is missing contexts: {sorted(missing_contexts)}")
    subset = scores.loc[scores["context"].isin((reference, target))].copy()
    identity_keys = [*METHOD_IDENTITY_KEYS, "contrast"]
    grouping = [*identity_keys, *EDGE_KEYS]
    global_base, global_index = _edge_group_base(subset, grouping)
    global_positions = pd.Series(
        np.arange(len(global_base), dtype=int),
        index=global_index,
    )
    frames: list[pd.DataFrame] = []
    for _, identity_group in subset.groupby(
        identity_keys,
        observed=True,
        sort=False,
    ):
        result = _paired_identity_effects(
            identity_group,
            grouping=grouping,
            reference=reference,
            target=target,
            min_pairs=min_pairs,
        )
        result_index = pd.MultiIndex.from_frame(result.loc[:, grouping])
        result["_row_position"] = global_positions.reindex(result_index).to_numpy(
            dtype=int
        )
        frames.append(result)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.sort_values("_row_position", kind="stable", ignore_index=True)
        .drop(columns="_row_position")
    )


def unpaired_edge_effects(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    min_subjects: int = 3,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Compute independent-group effects after subject-context aggregation.

    Multiple libraries from the same subject and context are averaged first.
    The target and reference means therefore weight biological subjects equally,
    regardless of how many libraries or regions each subject contributed.
    """

    if (
        isinstance(min_subjects, bool)
        or not isinstance(min_subjects, int)
        or min_subjects < 2
    ):
        raise ValueError("min_subjects must be an integer >= 2")
    if reference == target:
        raise ValueError("reference and target contexts must differ")
    scores = _normalize_subject_context(
        _select_contrast(
            _prepared_score_table(table, validated=validated), contrast
        )
    )
    scores = scores.loc[scores["context"].isin((reference, target))]
    identity_keys = [*METHOD_IDENTITY_KEYS, "contrast"]
    result_frames: list[pd.DataFrame] = []
    for _identity_values, identity_group in scores.groupby(
        identity_keys, observed=True, sort=False
    ):
        reference_subjects, target_subjects = _validate_unpaired_contexts(
            identity_group,
            reference=reference,
            target=target,
        )
        reference_order = tuple(
            identity_group.loc[
                identity_group["context"].eq(reference), "subject_id"
            ].drop_duplicates()
        )
        target_order = tuple(
            identity_group.loc[
                identity_group["context"].eq(target), "subject_id"
            ].drop_duplicates()
        )
        grouping = [*identity_keys, *EDGE_KEYS]
        base, edge_index = _edge_group_base(identity_group, grouping)
        usable = identity_group.loc[identity_group["comparison_eligible"]]
        subject_grouping = (*EDGE_KEYS, "subject_id", "context")
        subject_groupers = [usable[key] for key in subject_grouping]
        row_positions = pd.Series(
            np.arange(len(identity_group), dtype=int),
            index=identity_group.index,
        )
        subject_scores = usable.groupby(
            subject_groupers,
            observed=True,
            sort=False,
        )["comparison_strength"].mean()
        subject_first = row_positions.loc[usable.index].groupby(
            subject_groupers,
            observed=True,
            sort=False,
        ).min()

        reference_values, reference_order_values = _context_subject_values(
            subject_scores,
            subject_first,
            edge_index=edge_index,
            identity_keys=identity_keys,
            context=reference,
            subjects=reference_order,
            n_edges=len(base),
        )
        target_values, target_order_values = _context_subject_values(
            subject_scores,
            subject_first,
            edge_index=edge_index,
            identity_keys=identity_keys,
            context=target,
            subjects=target_order,
            n_edges=len(base),
        )
        n_reference, reference_means, _, _, reference_variances = (
            _rowwise_distribution(
                reference_values,
                order=reference_order_values,
            )
        )
        n_target, target_means, _, _, target_variances = _rowwise_distribution(
            target_values,
            order=target_order_values,
        )
        effects = target_means - reference_means
        standard_errors: np.ndarray = np.full(len(base), np.nan, dtype=float)
        replicated = (n_reference >= 2) & (n_target >= 2)
        pooled_denominator = n_reference + n_target - 2
        pooled_variance: np.ndarray = np.full(len(base), np.nan, dtype=float)
        standardized: np.ndarray = np.full(len(base), np.nan, dtype=float)
        for row in np.flatnonzero(replicated):
            n_reference_row = int(n_reference[row])
            n_target_row = int(n_target[row])
            standard_errors[row] = float(
                math.sqrt(
                    target_variances[row] / n_target_row
                    + reference_variances[row] / n_reference_row
                )
            )
            denominator = int(pooled_denominator[row])
            if denominator <= 0:
                continue
            variance = (
                (n_reference_row - 1) * reference_variances[row]
                + (n_target_row - 1) * target_variances[row]
            ) / denominator
            pooled_variance[row] = variance
            if np.isfinite(effects[row]) and np.isfinite(variance) and variance > 0:
                standardized[row] = effects[row] / math.sqrt(variance)

        sample_counts = (
            usable.groupby(
                [usable[key] for key in (*EDGE_KEYS, "context")],
                observed=True,
                sort=False,
            )["sample_id"]
            .nunique()
            .unstack("context")
            .reindex(edge_index.droplevel(identity_keys))
        )
        n_reference_samples = (
            sample_counts[reference].fillna(0).to_numpy(dtype=int)
            if reference in sample_counts
            else np.zeros(len(base), dtype=int)
        )
        n_target_samples = (
            sample_counts[target].fillna(0).to_numpy(dtype=int)
            if target in sample_counts
            else np.zeros(len(base), dtype=int)
        )
        status_summary = _edge_status_summary(identity_group, grouping).reindex(
            edge_index
        )
        estimable = (n_reference >= min_subjects) & (n_target >= min_subjects)

        result = base.copy()
        result["reference"] = reference
        result["target"] = target
        result["effect"] = effects
        result["effect_semantics"] = (
            "target_minus_reference_subject_mean_comparison_strength"
        )
        result["diagnostic_standard_error"] = standard_errors
        result["standardized_effect"] = standardized
        result["reference_mean"] = reference_means
        result["target_mean"] = target_means
        result["n_reference_subjects"] = n_reference
        result["n_target_subjects"] = n_target
        result["n_reference_subjects_total"] = len(reference_subjects)
        result["n_target_subjects_total"] = len(target_subjects)
        result["n_reference_samples"] = n_reference_samples
        result["n_target_samples"] = n_target_samples
        result["minimum_subjects_per_context"] = min_subjects
        result["aggregation"] = (
            "sample_mean_within_subject_context_then_equal_subject_mean"
        )
        result["design"] = "independent_subject_groups"
        for column in (
            "n_not_predicted_rows",
            "n_missing_rows",
            "n_resource_unavailable_rows",
            "n_cell_type_missing_rows",
        ):
            result[column] = status_summary[column].to_numpy(dtype=int)
        result["status"] = np.where(estimable, "exploratory", "not_estimable")
        result["reason_code"] = _unpaired_reason_codes(
            status_summary,
            n_reference=n_reference,
            n_target=n_target,
            min_subjects=min_subjects,
            estimable=estimable,
        )
        result_frames.append(result)
    return (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )


def _effect_top_keys(effect: pd.Series, *, k: int) -> set[tuple[object, ...]]:
    if effect.empty:
        return set()
    magnitude = effect.abs().sort_values(ascending=False, kind="stable")
    cutoff = float(magnitude.iloc[min(k, len(magnitude)) - 1])
    selected = magnitude.loc[magnitude >= cutoff]
    return set(selected.index.tolist())


def _empty_edge_series() -> pd.Series:
    index = pd.MultiIndex.from_arrays([[] for _ in EDGE_KEYS], names=list(EDGE_KEYS))
    return pd.Series(index=index, dtype=float)


def _family_loso_metrics(
    held: pd.Series,
    training: pd.Series,
    universe: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, object]:
    held_frame = held.rename("held_effect").reset_index()
    training_frame = training.rename("training_effect").reset_index()
    shared = held_frame.merge(
        training_frame,
        on=list(EDGE_KEYS),
        validate="one_to_one",
    )
    shared_family_rows = shared.groupby(
        list(FAMILY_KEYS), sort=False, observed=True
    ).indices
    rho_values: list[float] = []
    direction_values: list[float] = []
    jaccard_values: list[float] = []
    family_coverages: list[float] = []
    direction_edges = 0
    eligible_families = 0
    family_sizes = universe.groupby(
        list(FAMILY_KEYS), sort=False, observed=True
    ).size()
    for family_key, family_size_value in family_sizes.items():
        family = cast(tuple[object, object], family_key)
        family_rows = shared_family_rows.get(family)
        if family_rows is None:
            family_shared = shared.iloc[:0]
        else:
            family_shared = shared.iloc[family_rows]
        family_size = int(family_size_value)
        family_coverages.append(len(family_shared) / family_size)
        if family_size < 3:
            continue
        eligible_families += 1
        enough_edges = len(family_shared) >= 3
        constant = bool(
            enough_edges
            and (
                family_shared["held_effect"].nunique() < 2
                or family_shared["training_effect"].nunique() < 2
            )
        )
        if enough_edges and not constant:
            rho = spearmanr(
                family_shared["held_effect"],
                family_shared["training_effect"],
            ).statistic
            if np.isfinite(rho):
                rho_values.append(float(rho))

        nonzero = ~(
            family_shared["held_effect"].eq(0) & family_shared["training_effect"].eq(0)
        )
        n_direction = int(nonzero.sum())
        direction_edges += n_direction
        if n_direction:
            direction_values.append(
                float(
                    np.mean(
                        np.sign(family_shared.loc[nonzero, "held_effect"])
                        == np.sign(family_shared.loc[nonzero, "training_effect"])
                    )
                )
            )

        # The merged row index uniquely identifies an edge, so it is sufficient
        # for comparing the two top-k sets and avoids rebuilding edge indexes.
        held_family = family_shared["held_effect"]
        training_family = family_shared["training_effect"]
        held_top = _effect_top_keys(held_family, k=top_k)
        training_top = _effect_top_keys(training_family, k=top_k)
        top_union = held_top | training_top
        if top_union:
            jaccard_values.append(len(held_top & training_top) / len(top_union))

    return {
        "shared_edges": len(shared),
        "n_eligible_families": eligible_families,
        "n_estimable_families": len(rho_values),
        "effect_spearman": (float(np.mean(rho_values)) if rho_values else math.nan),
        "direction_comparable_edges": direction_edges,
        "direction_agreement": (
            float(np.mean(direction_values)) if direction_values else math.nan
        ),
        "top_k_jaccard": (
            float(np.mean(jaccard_values)) if jaccard_values else math.nan
        ),
        "macro_family_shared_edge_coverage": (
            float(np.mean(family_coverages)) if family_coverages else math.nan
        ),
    }


def _identity_rng(
    random_seed: int, identity_values: tuple[object, ...]
) -> np.random.Generator:
    payload = json.dumps(
        [random_seed, *(str(value) for value in identity_values)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


def _complete_unpaired_strength_matrices(
    group: pd.DataFrame,
    *,
    reference: str,
    target: str,
    reference_subjects: tuple[str, ...],
    target_subjects: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = _frozen_eligible_universe(group)
    subject_edges = _subject_edge_strengths(group)
    expected_subjects = len(reference_subjects) + len(target_subjects)
    if subject_edges.empty or expected_subjects == 0:
        complete_index = pd.MultiIndex.from_arrays(
            [[] for _ in EDGE_KEYS], names=list(EDGE_KEYS)
        )
    else:
        support = subject_edges.groupby(level=list(EDGE_KEYS), observed=True).size()
        complete_edges = support.loc[support.eq(expected_subjects)].index
        complete_index = pd.MultiIndex.from_tuples(
            list(complete_edges), names=list(EDGE_KEYS)
        )

    def context_matrix(context: str, subjects: tuple[str, ...]) -> pd.DataFrame:
        if subject_edges.empty or len(complete_index) == 0:
            return pd.DataFrame(
                index=pd.Index(subjects, name="subject_id"),
                columns=complete_index,
                dtype=float,
            )
        context_scores = subject_edges.xs(context, level="context")
        matrix = context_scores.unstack(list(EDGE_KEYS))
        return matrix.reindex(index=subjects, columns=complete_index)

    return (
        context_matrix(reference, reference_subjects),
        context_matrix(target, target_subjects),
        universe,
    )


def unpaired_differential_split_half_reproducibility(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    n_repeats: int = 200,
    min_subjects_per_half: int = 2,
    top_k: int = 100,
    random_seed: int = 0,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Estimate independent-group differential-rank split-half stability.

    Each repeat independently splits reference and target subjects into two
    halves, computes a target-minus-reference edge effect in both halves, and
    macro-averages within-family Spearman correlations. The empirical interval
    describes repeated-split stability; it is not an inferential confidence
    interval for a biological group effect.
    """

    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 2:
        raise ValueError("n_repeats must be an integer >= 2")
    if (
        isinstance(min_subjects_per_half, bool)
        or not isinstance(min_subjects_per_half, int)
        or min_subjects_per_half < 2
    ):
        raise ValueError("min_subjects_per_half must be an integer >= 2")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")
    if reference == target:
        raise ValueError("reference and target contexts must differ")
    scores = _normalize_subject_context(
        _select_contrast(
            _prepared_score_table(table, validated=validated), contrast
        )
    )
    scores = scores.loc[scores["context"].isin((reference, target))]
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    rows: list[dict[str, object]] = []
    for identity_values, group in scores.groupby(grouping, observed=True, sort=False):
        reference_subjects, target_subjects = _validate_unpaired_contexts(
            group,
            reference=reference,
            target=target,
        )
        reference_matrix, target_matrix, universe = (
            _complete_unpaired_strength_matrices(
                group,
                reference=reference,
                target=target,
                reference_subjects=reference_subjects,
                target_subjects=target_subjects,
            )
        )
        reference_half_size = len(reference_subjects) // 2
        target_half_size = len(target_subjects) // 2
        enough_subjects = (
            reference_half_size >= min_subjects_per_half
            and len(reference_subjects) - reference_half_size >= min_subjects_per_half
            and target_half_size >= min_subjects_per_half
            and len(target_subjects) - target_half_size >= min_subjects_per_half
        )
        repeat_spearman: list[float] = []
        repeat_direction: list[float] = []
        repeat_family_counts: list[int] = []
        split_tokens: list[str] = []
        rng = _identity_rng(random_seed, identity_values)
        if enough_subjects and reference_matrix.shape[1] > 0:
            reference_array = reference_matrix.to_numpy(dtype=float)
            target_array = target_matrix.to_numpy(dtype=float)
            for _ in range(n_repeats):
                reference_order = rng.permutation(len(reference_subjects))
                target_order = rng.permutation(len(target_subjects))
                reference_left = reference_order[:reference_half_size]
                reference_right = reference_order[reference_half_size:]
                target_left = target_order[:target_half_size]
                target_right = target_order[target_half_size:]
                left_effect = pd.Series(
                    target_array[target_left].mean(axis=0)
                    - reference_array[reference_left].mean(axis=0),
                    index=reference_matrix.columns,
                    dtype=float,
                )
                right_effect = pd.Series(
                    target_array[target_right].mean(axis=0)
                    - reference_array[reference_right].mean(axis=0),
                    index=reference_matrix.columns,
                    dtype=float,
                )
                metrics = _family_loso_metrics(
                    left_effect,
                    right_effect,
                    universe,
                    top_k=top_k,
                )
                rho = float(str(metrics["effect_spearman"]))
                if np.isfinite(rho):
                    repeat_spearman.append(rho)
                    repeat_family_counts.append(
                        int(str(metrics["n_estimable_families"]))
                    )
                    direction = float(str(metrics["direction_agreement"]))
                    if np.isfinite(direction):
                        repeat_direction.append(direction)
                split_tokens.append(
                    json.dumps(
                        {
                            "reference_left": sorted(
                                reference_subjects[index] for index in reference_left
                            ),
                            "target_left": sorted(
                                target_subjects[index] for index in target_left
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        valid_repeats = len(repeat_spearman)
        estimable = valid_repeats >= 2
        if estimable:
            spearman_values = np.asarray(repeat_spearman, dtype=float)
            quantiles = np.asarray(
                np.quantile(spearman_values, [0.025, 0.975]), dtype=float
            )
            estimate = float(spearman_values.mean())
            ci_lower = float(quantiles[0])
            ci_upper = float(quantiles[1])
        else:
            estimate = math.nan
            ci_lower = math.nan
            ci_upper = math.nan
        if estimable:
            reason = None
        elif not enough_subjects:
            reason = "insufficient_subjects_for_stratified_halves"
        elif reference_matrix.shape[1] == 0:
            reason = "no_complete_subject_edges"
        else:
            reason = "fewer_than_two_valid_split_repeats"
        identity = dict(zip(grouping, identity_values, strict=True))
        identity.update(
            {
                "endpoint": (
                    "family_macro_unpaired_differential_rank_repeated_split_half"
                ),
                "reference": reference,
                "target": target,
                "estimate": estimate,
                "effect_spearman": estimate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": 0.95,
                "interval_type": "empirical_repeated_split_quantiles",
                "interval_semantics": (
                    "stability_diagnostic_not_biological_effect_inference"
                ),
                "n_repeats_requested": n_repeats,
                "n_repeats_valid": valid_repeats,
                "valid_repeat_fraction": valid_repeats / n_repeats,
                "n_unique_splits": len(set(split_tokens)),
                "split_assignment_sha256": (
                    hashlib.sha256("\n".join(split_tokens).encode("utf-8")).hexdigest()
                    if split_tokens
                    else None
                ),
                "random_seed": random_seed,
                "n_reference_subjects": len(reference_subjects),
                "n_target_subjects": len(target_subjects),
                "reference_half_size_minimum": reference_half_size,
                "reference_half_size_maximum": (
                    len(reference_subjects) - reference_half_size
                ),
                "target_half_size_minimum": target_half_size,
                "target_half_size_maximum": len(target_subjects) - target_half_size,
                "minimum_subjects_per_half": min_subjects_per_half,
                "frozen_eligible_edges": len(universe),
                "complete_subject_edges": reference_matrix.shape[1],
                "complete_subject_edge_coverage": (
                    reference_matrix.shape[1] / len(universe)
                    if len(universe)
                    else math.nan
                ),
                "median_estimable_families": (
                    float(np.median(repeat_family_counts))
                    if repeat_family_counts
                    else math.nan
                ),
                "mean_direction_agreement": (
                    float(np.mean(repeat_direction)) if repeat_direction else math.nan
                ),
                "n_direction_valid_repeats": len(repeat_direction),
                "top_k": top_k,
                "split_unit": "subject_stratified_within_context",
                "aggregation": "equal_families_then_equal_repeated_splits",
                "status": "observed" if estimable else "not_estimable",
                "reason_code": reason,
            }
        )
        rows.append(identity)
    return pd.DataFrame(rows)


def unpaired_leave_one_subject_influence(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    min_remaining_subjects: int = 2,
    top_k: int = 100,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Diagnose independent-group sensitivity to each subject's removal.

    This is an influence analysis against the full-cohort effect vector. It is
    intentionally not labelled or interpreted as independent replication.
    """

    if (
        isinstance(min_remaining_subjects, bool)
        or not isinstance(min_remaining_subjects, int)
        or min_remaining_subjects < 2
    ):
        raise ValueError("min_remaining_subjects must be an integer >= 2")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    if reference == target:
        raise ValueError("reference and target contexts must differ")
    scores = _normalize_subject_context(
        _select_contrast(
            _prepared_score_table(table, validated=validated), contrast
        )
    )
    scores = scores.loc[scores["context"].isin((reference, target))]
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    rows: list[dict[str, object]] = []
    for identity_values, group in scores.groupby(grouping, observed=True, sort=False):
        reference_subjects, target_subjects = _validate_unpaired_contexts(
            group,
            reference=reference,
            target=target,
        )
        reference_matrix, target_matrix, universe = (
            _complete_unpaired_strength_matrices(
                group,
                reference=reference,
                target=target,
                reference_subjects=reference_subjects,
                target_subjects=target_subjects,
            )
        )
        full_effect = pd.Series(
            target_matrix.mean(axis=0).to_numpy(dtype=float)
            - reference_matrix.mean(axis=0).to_numpy(dtype=float),
            index=reference_matrix.columns,
            dtype=float,
        )
        identity = dict(zip(grouping, identity_values, strict=True))
        for excluded_context, subjects in (
            (reference, reference_subjects),
            (target, target_subjects),
        ):
            for excluded_subject in subjects:
                remaining_reference = reference_matrix.drop(
                    index=excluded_subject if excluded_context == reference else [],
                    errors="ignore",
                )
                remaining_target = target_matrix.drop(
                    index=excluded_subject if excluded_context == target else [],
                    errors="ignore",
                )
                enough_subjects = (
                    len(remaining_reference) >= min_remaining_subjects
                    and len(remaining_target) >= min_remaining_subjects
                )
                has_edges = reference_matrix.shape[1] > 0
                estimable = enough_subjects and has_edges
                if estimable:
                    leave_one_out_effect = pd.Series(
                        remaining_target.mean(axis=0).to_numpy(dtype=float)
                        - remaining_reference.mean(axis=0).to_numpy(dtype=float),
                        index=reference_matrix.columns,
                        dtype=float,
                    )
                    difference = leave_one_out_effect - full_effect
                    metrics = _family_loso_metrics(
                        full_effect,
                        leave_one_out_effect,
                        universe,
                        top_k=top_k,
                    )
                    nonzero = ~(full_effect.eq(0) & leave_one_out_effect.eq(0))
                    direction_edges = int(nonzero.sum())
                    direction_agreement = (
                        float(
                            np.mean(
                                np.sign(full_effect.loc[nonzero])
                                == np.sign(leave_one_out_effect.loc[nonzero])
                            )
                        )
                        if direction_edges
                        else math.nan
                    )
                    rank_similarity = float(str(metrics["effect_spearman"]))
                    mean_absolute_change = float(difference.abs().mean())
                    max_absolute_change = float(difference.abs().max())
                    root_mean_square_change = float(
                        math.sqrt(np.mean(np.square(difference.to_numpy(dtype=float))))
                    )
                    n_changed_edges = int(difference.ne(0).sum())
                    n_estimable_families = int(str(metrics["n_estimable_families"]))
                else:
                    direction_edges = 0
                    direction_agreement = math.nan
                    rank_similarity = math.nan
                    mean_absolute_change = math.nan
                    max_absolute_change = math.nan
                    root_mean_square_change = math.nan
                    n_changed_edges = 0
                    n_estimable_families = 0
                row = identity.copy()
                row.update(
                    {
                        "diagnostic_name": "leave_one_subject_influence",
                        "interpretation": (
                            "influence_diagnostic_not_independent_replication"
                        ),
                        "reference": reference,
                        "target": target,
                        "excluded_subject": excluded_subject,
                        "excluded_context": excluded_context,
                        "n_reference_subjects_remaining": len(remaining_reference),
                        "n_target_subjects_remaining": len(remaining_target),
                        "minimum_remaining_subjects": min_remaining_subjects,
                        "frozen_eligible_edges": len(universe),
                        "complete_subject_edges": reference_matrix.shape[1],
                        "family_macro_rank_similarity_to_full": rank_similarity,
                        "n_estimable_families": n_estimable_families,
                        "direction_comparable_edges": direction_edges,
                        "direction_agreement_to_full": direction_agreement,
                        "mean_absolute_effect_change": mean_absolute_change,
                        "maximum_absolute_effect_change": max_absolute_change,
                        "root_mean_square_effect_change": root_mean_square_change,
                        "n_changed_edges": n_changed_edges,
                        "top_k": top_k,
                        "status": "descriptive" if estimable else "not_estimable",
                        "reason_code": (
                            None
                            if estimable
                            else "insufficient_subjects_after_exclusion"
                            if not enough_subjects
                            else "no_complete_subject_edges"
                        ),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def paired_differential_loso_reproducibility(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    min_subjects: int = 3,
    top_k: int = 100,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Estimate family-macro paired differential LOSO rank reproducibility.

    Multiple samples in one subject/context are averaged before leave-one-subject-
    out folds are formed. This prevents regions or technical replicates from being
    counted as independent biological replicates. Training effects must be
    available in every remaining paired subject, preventing a favorable inner
    complete-case universe from changing edge by edge. Spearman and top-k metrics
    are computed within sender-receiver families and macro-averaged so large
    families cannot dominate the primary endpoint.
    """

    if (
        isinstance(min_subjects, bool)
        or not isinstance(min_subjects, int)
        or min_subjects < 3
    ):
        raise ValueError("min_subjects must be an integer >= 3")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    if reference == target:
        raise ValueError("reference and target contexts must differ")
    scores = _normalize_subject_context(
        _select_contrast(
            _prepared_score_table(table, validated=validated), contrast
        )
    )
    available_contexts = set(scores["context"])
    missing_contexts = {reference, target}.difference(available_contexts)
    if missing_contexts:
        raise ValueError(f"score table is missing contexts: {sorted(missing_contexts)}")
    scores = scores.loc[scores["context"].isin((reference, target))]
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    rows: list[dict[str, object]] = []
    for keys, group in scores.groupby(grouping, observed=True, sort=False):
        edge_status = group.groupby(list(EDGE_KEYS), sort=False, observed=True)[
            "status"
        ]
        universe_edges = [
            edge
            for edge, statuses in edge_status
            if not statuses.eq("resource_unavailable").all()
        ]
        universe = pd.DataFrame(universe_edges, columns=EDGE_KEYS)
        usable = group.loc[group["comparison_eligible"]]
        subject_scores = (
            usable.groupby(
                ["subject_id", "context", *EDGE_KEYS],
                observed=True,
                sort=False,
            )["comparison_strength"]
            .mean()
            .unstack("context")
        )
        if reference in subject_scores and target in subject_scores:
            complete = subject_scores[[reference, target]].dropna()
            paired_effect = complete[target] - complete[reference]
        else:
            empty_index = pd.MultiIndex.from_arrays(
                [[] for _ in range(1 + len(EDGE_KEYS))],
                names=["subject_id", *EDGE_KEYS],
            )
            paired_effect = pd.Series(index=empty_index, dtype=float)
        all_subjects = tuple(sorted(group["subject_id"].unique()))
        context_support = group.loc[:, ["subject_id", "context"]].drop_duplicates()
        reference_subjects = frozenset(
            context_support.loc[context_support["context"].eq(reference), "subject_id"]
        )
        target_subjects = frozenset(
            context_support.loc[context_support["context"].eq(target), "subject_id"]
        )
        paired_subjects = reference_subjects & target_subjects
        for held_out in all_subjects:
            identity = dict(zip(grouping, keys, strict=True))
            held = (
                paired_effect.xs(held_out, level="subject_id")
                if held_out in paired_subjects
                and held_out in paired_effect.index.get_level_values("subject_id")
                else _empty_edge_series()
            )
            training_subjects = paired_subjects - {held_out}
            training = paired_effect.loc[
                paired_effect.index.get_level_values("subject_id").isin(
                    training_subjects
                )
            ]
            if training.empty or not training_subjects:
                train_mean = _empty_edge_series()
                support = _empty_edge_series()
            else:
                edge_levels: Sequence[str] = EDGE_KEYS
                support = training.groupby(level=edge_levels, observed=True).size()
                complete_training_edges = support.loc[
                    support.eq(len(training_subjects))
                ].index
                train_mean = (
                    training.groupby(level=edge_levels, observed=True)
                    .mean()
                    .loc[complete_training_edges]
                )
            family_metrics = _family_loso_metrics(
                held.astype(float),
                train_mean.astype(float),
                universe,
                top_k=top_k,
            )
            enough_subjects = (
                held_out in paired_subjects and len(paired_subjects) >= min_subjects
            )
            estimable = enough_subjects and bool(family_metrics["n_estimable_families"])
            reason = (
                None
                if estimable
                else "insufficient_paired_subjects"
                if not enough_subjects
                else "no_eligible_sender_receiver_families"
                if not family_metrics["n_eligible_families"]
                else "insufficient_or_constant_family_ranks"
            )
            shared_edges = int(str(family_metrics["shared_edges"]))
            identity.update(
                {
                    "reference": reference,
                    "target": target,
                    "held_out_subject": held_out,
                    "n_paired_subjects": len(paired_subjects),
                    "n_training_subjects": len(training_subjects),
                    "frozen_eligible_edges": len(universe),
                    "held_out_complete_edges": len(held),
                    "training_complete_edges": len(train_mean),
                    "shared_edges": shared_edges,
                    "held_out_edge_coverage": (
                        len(held) / len(universe) if len(universe) else math.nan
                    ),
                    "training_edge_coverage": (
                        len(train_mean) / len(universe) if len(universe) else math.nan
                    ),
                    "shared_edge_coverage": (
                        shared_edges / len(universe) if len(universe) else math.nan
                    ),
                    "minimum_training_edge_support": (
                        int(support.loc[train_mean.index].min())
                        if len(train_mean)
                        else 0
                    ),
                    **family_metrics,
                    "top_k": top_k,
                    "status": "observed" if estimable else "not_estimable",
                    "reason_code": reason,
                }
            )
            rows.append(identity)
    return pd.DataFrame(rows)


def _validate_loso_endpoint_table(loso: pd.DataFrame) -> pd.DataFrame:
    required = {
        *METHOD_IDENTITY_KEYS,
        "contrast",
        "held_out_subject",
        "effect_spearman",
        "status",
    }
    missing = required.difference(loso.columns)
    if missing:
        raise ValueError(f"LOSO table is missing columns: {sorted(missing)}")
    key = [*METHOD_IDENTITY_KEYS, "contrast", "held_out_subject"]
    if loso.duplicated(key).any():
        raise ValueError("LOSO table contains duplicate held-out-subject folds")
    invalid_statuses = set(loso["status"].astype(str)).difference(
        {"observed", "not_estimable"}
    )
    if invalid_statuses:
        raise ValueError(f"LOSO table has invalid statuses: {sorted(invalid_statuses)}")
    values = pd.to_numeric(loso["effect_spearman"], errors="coerce")
    supplied = loso["effect_spearman"].notna()
    observed = loso["status"].eq("observed")
    if (
        (supplied & values.isna()).any()
        or np.isinf(values.fillna(0.0)).any()
        or (values.dropna().abs() > 1).any()
        or values.loc[observed].isna().any()
    ):
        raise ValueError("effect_spearman values must be finite in [-1, 1] when set")
    result = loso.copy(deep=True)
    result["effect_spearman"] = values
    return result


def summarize_loso_primary_endpoint(
    loso: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Macro-average LOSO folds by subject with a subject-block bootstrap CI."""

    loso = _validate_loso_endpoint_table(loso)
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, int)
        or n_bootstrap < 1
    ):
        raise ValueError("n_bootstrap must be an integer >= 1")
    key = [*METHOD_IDENTITY_KEYS, "contrast", "held_out_subject"]
    grouping = [*METHOD_IDENTITY_KEYS, "contrast"]
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    ordered = loso.sort_values(key, kind="stable")
    for keys, group in ordered.groupby(grouping, sort=False, observed=True):
        values = pd.to_numeric(
            group.loc[group["status"].eq("observed"), "effect_spearman"],
            errors="coerce",
        )
        values = values.loc[np.isfinite(values)].to_numpy(dtype=float)
        estimable = len(values) >= 2
        if estimable:
            bootstrap: np.ndarray = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                bootstrap[index] = float(
                    rng.choice(values, size=len(values), replace=True).mean()
                )
            estimate = float(values.mean())
            quantiles = np.asarray(np.quantile(bootstrap, [0.025, 0.975]), dtype=float)
            lower = float(quantiles[0])
            upper = float(quantiles[1])
        else:
            estimate = math.nan
            lower = math.nan
            upper = math.nan
        identity = dict(zip(grouping, keys, strict=True))
        identity.update(
            {
                "endpoint": "family_macro_within_subject_differential_rank_loso",
                "estimate": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": 0.95,
                "bootstrap_unit": "held_out_subject",
                "n_bootstrap": n_bootstrap,
                "n_subjects_total": int(group["held_out_subject"].nunique()),
                "n_subjects_estimable": len(values),
                "aggregation": "equal_families_then_equal_subjects",
                "status": "observed" if estimable else "not_estimable",
                "reason_code": (
                    None if estimable else "fewer_than_two_estimable_subject_folds"
                ),
            }
        )
        rows.append(identity)
    return pd.DataFrame(rows)


def aggregate_loso_primary_endpoint(
    loso: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Equal-weight prespecified dataset means with a nested block bootstrap."""

    loso = _validate_loso_endpoint_table(loso)
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, int)
        or n_bootstrap < 1
    ):
        raise ValueError("n_bootstrap must be an integer >= 1")
    system_keys = (
        "method",
        "method_version",
        "analysis_track",
        "resource",
        "resource_version",
        "resource_mode",
        "score_semantics",
    )
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for keys, system in loso.groupby(list(system_keys), sort=False, observed=True):
        dataset_values: dict[str, np.ndarray] = {}
        dataset_metadata: list[dict[str, object]] = []
        for dataset, group in system.groupby("dataset", sort=False, observed=True):
            if group["contrast"].nunique() != 1 or group["universe_id"].nunique() != 1:
                raise ValueError(
                    "cross-dataset aggregation requires one prespecified contrast "
                    "and universe per method-dataset"
                )
            values = pd.to_numeric(
                group.loc[group["status"].eq("observed"), "effect_spearman"],
                errors="coerce",
            )
            values = values.loc[np.isfinite(values)].to_numpy(dtype=float)
            if len(values):
                dataset_values[str(dataset)] = values
                dataset_metadata.append(
                    {
                        "dataset": str(dataset),
                        "contrast": str(group["contrast"].iloc[0]),
                        "universe_id": str(group["universe_id"].iloc[0]),
                    }
                )
        dataset_names = tuple(sorted(dataset_values))
        estimable = len(dataset_names) >= 2
        if estimable:
            dataset_means = np.asarray(
                [dataset_values[name].mean() for name in dataset_names], dtype=float
            )
            estimate = float(dataset_means.mean())
            bootstrap: np.ndarray = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                sampled_datasets = rng.choice(
                    dataset_names, size=len(dataset_names), replace=True
                )
                sampled_means = [
                    rng.choice(
                        dataset_values[str(dataset)],
                        size=len(dataset_values[str(dataset)]),
                        replace=True,
                    ).mean()
                    for dataset in sampled_datasets
                ]
                bootstrap[index] = float(np.mean(sampled_means))
            quantiles = np.asarray(np.quantile(bootstrap, [0.025, 0.975]), dtype=float)
            lower = float(quantiles[0])
            upper = float(quantiles[1])
        else:
            estimate = math.nan
            lower = math.nan
            upper = math.nan
        row = dict(zip(system_keys, keys, strict=True))
        row.update(
            {
                "endpoint": "cross_dataset_family_macro_differential_rank_loso",
                "estimate": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": 0.95,
                "bootstrap_unit": "dataset_then_subject",
                "n_bootstrap": n_bootstrap,
                "n_datasets": len(dataset_names),
                "n_subjects_estimable": int(
                    sum(len(dataset_values[name]) for name in dataset_names)
                ),
                "dataset_contract_json": json.dumps(
                    sorted(dataset_metadata, key=lambda item: str(item["dataset"])),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "aggregation": (
                    "equal_families_then_equal_subjects_then_equal_datasets"
                ),
                "status": "observed" if estimable else "not_estimable",
                "reason_code": (
                    None if estimable else "fewer_than_two_estimable_datasets"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_rank_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    ranks = pd.Series(scores).rank(ascending=True, method="average").to_numpy()
    u_statistic = float(ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
    auroc = u_statistic / (n_positive * n_negative)

    ordered = pd.DataFrame({"label": labels, "score": scores}).sort_values(
        "score", ascending=False, kind="stable"
    )
    grouped = ordered.groupby("score", sort=False, observed=True)["label"].agg(
        ["sum", "count"]
    )
    true_positive = grouped["sum"].cumsum().to_numpy(dtype=float)
    predicted_positive = grouped["count"].cumsum().to_numpy(dtype=float)
    recall = true_positive / n_positive
    precision = true_positive / predicted_positive
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    average_precision = float(np.sum((recall - previous_recall) * precision))
    return float(auroc), average_precision


def synthetic_edge_truth_metrics(
    table: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k: int = 100,
    contrast: str | None = None,
    validated: bool = False,
) -> pd.DataFrame:
    """Evaluate edge truth only for explicitly synthetic/known-truth scenarios."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    truth_required = {
        "dataset",
        "contrast",
        "universe_id",
        *EDGE_KEYS,
        "is_positive",
        "truth_scope",
    }
    missing = truth_required.difference(truth.columns)
    if missing:
        raise ValueError(f"truth table is missing columns: {sorted(missing)}")
    invalid_scopes = set(truth["truth_scope"].astype(str)).difference(
        SYNTHETIC_TRUTH_SCOPES
    )
    if invalid_scopes:
        raise ValueError(
            "AUROC/AUPRC are forbidden without synthetic or perturbation truth; "
            f"invalid truth_scope={sorted(invalid_scopes)}"
        )
    truth_key = ["dataset", "contrast", "universe_id", *EDGE_KEYS]
    if truth.duplicated(truth_key).any():
        raise ValueError("truth table contains duplicate frozen-universe edges")
    labels = pd.to_numeric(truth["is_positive"], errors="coerce")
    if (
        labels.isna().any()
        or (labels % 1 != 0).any()
        or not set(labels.astype(int)).issubset({0, 1})
    ):
        raise ValueError("is_positive must contain only binary 0/1 labels")
    truth_table = truth.copy(deep=True)
    truth_table["is_positive"] = labels.astype(int)

    scores = _select_contrast(
        _prepared_score_table(table, validated=validated), contrast
    )
    grouping = [*METHOD_IDENTITY_KEYS, "contrast", "sample_id"]
    rows: list[dict[str, object]] = []
    for keys, sample in scores.groupby(grouping, sort=False, observed=True):
        dataset = str(sample["dataset"].iloc[0])
        selected_truth = truth_table.loc[
            truth_table["dataset"].astype(str).eq(dataset)
            & truth_table["contrast"].eq(sample["contrast"].iloc[0])
            & truth_table["universe_id"].eq(sample["universe_id"].iloc[0])
        ]
        score_edges = set(sample.loc[:, EDGE_KEYS].itertuples(index=False, name=None))
        truth_edges = set(
            selected_truth.loc[:, EDGE_KEYS].itertuples(index=False, name=None)
        )
        if score_edges != truth_edges:
            raise ValueError(
                "truth must exactly match the frozen score universe before scoring"
            )
        merged = sample.merge(
            selected_truth[[*EDGE_KEYS, "is_positive", "truth_scope"]],
            on=list(EDGE_KEYS),
            validate="one_to_one",
        )
        comparable = merged.loc[merged["comparison_eligible"]].copy()
        y_true = comparable["is_positive"].to_numpy(dtype=int)
        y_score = comparable["comparison_strength"].to_numpy(dtype=float)
        n_positive = int((y_true == 1).sum())
        n_negative = int((y_true == 0).sum())
        estimable = n_positive > 0 and n_negative > 0
        if estimable:
            auroc, average_precision = _binary_rank_metrics(y_true, y_score)
            selected_keys = _top_keys(comparable, k=top_k)
            positive_keys = set(
                comparable.loc[comparable["is_positive"].eq(1), EDGE_KEYS].itertuples(
                    index=False, name=None
                )
            )
            true_positive = len(selected_keys & positive_keys)
            false_positive = len(selected_keys - positive_keys)
            false_negative = len(positive_keys - selected_keys)
            true_negative = (
                len(comparable) - true_positive - false_positive - false_negative
            )
            precision = (
                true_positive / len(selected_keys) if selected_keys else math.nan
            )
            recall = true_positive / n_positive
            f1 = (
                2 * precision * recall / (precision + recall)
                if np.isfinite(precision) and precision + recall > 0
                else math.nan
            )
            mcc_denominator = math.sqrt(
                (true_positive + false_positive)
                * (true_positive + false_negative)
                * (true_negative + false_positive)
                * (true_negative + false_negative)
            )
            mcc = (
                (true_positive * true_negative - false_positive * false_negative)
                / mcc_denominator
                if mcc_denominator
                else math.nan
            )
        else:
            auroc = math.nan
            average_precision = math.nan
            selected_keys = set()
            precision = math.nan
            recall = math.nan
            f1 = math.nan
            mcc = math.nan
        identity = dict(zip(grouping, keys, strict=True))
        identity.update(
            {
                "subject_id": str(_one_value(sample["subject_id"], label="subject_id")),
                "context": str(_one_value(sample["context"], label="context")),
                "truth_scope": str(
                    _one_value(selected_truth["truth_scope"], label="truth_scope")
                ),
                "truth_universe_edges": len(merged),
                "comparison_eligible_edges": len(comparable),
                "truth_coverage_fraction": len(comparable) / len(merged),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "auroc": auroc,
                "average_precision": average_precision,
                "top_k": top_k,
                "top_k_realized": len(selected_keys),
                "top_k_precision": precision,
                "top_k_recall": recall,
                "top_k_f1": f1,
                "top_k_mcc": mcc,
                "status": "observed" if estimable else "not_estimable",
                "reason_code": None if estimable else "truth_has_single_class",
            }
        )
        rows.append(identity)
    return pd.DataFrame(rows)


def summarize_run_performance(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runtime, memory, output size, failures, and determinism audits."""

    required = {
        *PERFORMANCE_IDENTITY_KEYS,
        "run_id",
        "status",
        "wall_time_seconds",
        "peak_rss_mb",
        "output_bytes",
        "threads",
    }
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"run table is missing columns: {sorted(missing)}")
    if runs.duplicated([*PERFORMANCE_IDENTITY_KEYS, "run_id"]).any():
        raise ValueError("run table contains duplicate method run IDs")
    allowed_statuses = {"complete", "failed", "not_supported", "skipped"}
    invalid_statuses = set(runs["status"].astype(str)).difference(allowed_statuses)
    if invalid_statuses:
        raise ValueError(f"run table has invalid statuses: {sorted(invalid_statuses)}")
    invalid_modes = set(runs["resource_mode"].astype(str)).difference(RESOURCE_MODES)
    if invalid_modes:
        raise ValueError(
            f"resource_mode contains invalid arms: {sorted(invalid_modes)}"
        )
    invalid_tracks = set(runs["analysis_track"].astype(str)).difference(
        {"lr_stlr", "ligand_target_program"}
    )
    if invalid_tracks:
        raise ValueError(
            f"run table contains invalid analysis tracks: {sorted(invalid_tracks)}"
        )
    numeric_fields = ("wall_time_seconds", "peak_rss_mb", "output_bytes", "threads")
    prepared = runs.copy(deep=True)
    for field in numeric_fields:
        raw_supplied = prepared[field].notna()
        numeric = pd.to_numeric(prepared[field], errors="coerce")
        if (raw_supplied & numeric.isna()).any() or np.isinf(numeric.fillna(0.0)).any():
            raise ValueError(f"{field} must be finite numeric when supplied")
        prepared[field] = numeric
        supplied = prepared[field].notna()
        if (prepared.loc[supplied, field] < 0).any():
            raise ValueError(f"{field} must be non-negative")
    if (prepared.loc[prepared["threads"].notna(), "threads"] < 1).any():
        raise ValueError("threads must be >= 1 when supplied")
    if (prepared.loc[prepared["threads"].notna(), "threads"] % 1 != 0).any():
        raise ValueError("threads must be integral when supplied")
    complete = prepared["status"].eq("complete")
    if prepared.loc[complete, "wall_time_seconds"].isna().any():
        raise ValueError("complete runs require wall_time_seconds")

    rows: list[dict[str, object]] = []
    for keys, group in prepared.groupby(
        list(PERFORMANCE_IDENTITY_KEYS), sort=False, observed=True
    ):
        completed = group.loc[group["status"].eq("complete")]
        repeated_determinism_groups = 0
        deterministic_output: bool | None = None
        if {"determinism_key", "output_sha256"}.issubset(group.columns):
            repeated: list[bool] = []
            for _, repeated_group in completed.dropna(
                subset=["determinism_key", "output_sha256"]
            ).groupby("determinism_key", sort=False, observed=True):
                if len(repeated_group) >= 2:
                    repeated_determinism_groups += 1
                    repeated.append(repeated_group["output_sha256"].nunique() == 1)
            if repeated:
                deterministic_output = all(repeated)
        row = dict(zip(PERFORMANCE_IDENTITY_KEYS, keys, strict=True))
        row.update(
            {
                "n_runs": len(group),
                "n_complete": len(completed),
                "n_failed": int(group["status"].eq("failed").sum()),
                "n_not_supported": int(group["status"].eq("not_supported").sum()),
                "success_rate": len(completed) / len(group),
                "median_wall_time_seconds": (
                    float(completed["wall_time_seconds"].median())
                    if len(completed)
                    else math.nan
                ),
                "median_peak_rss_mb": (
                    float(completed["peak_rss_mb"].median())
                    if completed["peak_rss_mb"].notna().any()
                    else math.nan
                ),
                "median_output_bytes": (
                    float(completed["output_bytes"].median())
                    if completed["output_bytes"].notna().any()
                    else math.nan
                ),
                "threads_minimum": (
                    int(completed["threads"].min())
                    if completed["threads"].notna().any()
                    else math.nan
                ),
                "threads_maximum": (
                    int(completed["threads"].max())
                    if completed["threads"].notna().any()
                    else math.nan
                ),
                "repeated_determinism_groups": repeated_determinism_groups,
                "deterministic_output": deterministic_output,
                "status": "observed" if len(completed) else "not_estimable",
                "reason_code": None if len(completed) else "no_successful_runs",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cross_method_concordance(
    effects: pd.DataFrame,
    *,
    minimum_shared_edges: int = 20,
) -> pd.DataFrame:
    """Compare estimable effect ranks for pairs of versioned method systems."""

    required = {
        *METHOD_IDENTITY_KEYS,
        "contrast",
        *EDGE_KEYS,
        "effect",
        "status",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"effect table is missing columns: {sorted(missing)}")
    if (
        isinstance(minimum_shared_edges, bool)
        or not isinstance(minimum_shared_edges, int)
        or minimum_shared_edges < 3
    ):
        raise ValueError("minimum_shared_edges must be an integer >= 3")
    metadata = list(required - {"effect"})
    if effects[metadata].isna().any().any():
        raise ValueError("effect identifiers and statuses must not be missing")
    if set(effects["analysis_track"].astype(str)) != {"lr_stlr"}:
        raise ValueError("LR concordance accepts only analysis_track='lr_stlr'")
    allowed_effect_statuses = EFFECT_ESTIMABLE_STATUSES | frozenset(
        {"not_estimable", "failed", "not_supported"}
    )
    invalid_statuses = set(effects["status"].astype(str)).difference(
        allowed_effect_statuses
    )
    if invalid_statuses:
        raise ValueError(
            f"effect table contains invalid statuses: {sorted(invalid_statuses)}"
        )
    numeric_effect = pd.to_numeric(effects["effect"], errors="coerce")
    supplied = effects["effect"].notna()
    if (
        (supplied & numeric_effect.isna()) | np.isinf(numeric_effect.fillna(0.0))
    ).any():
        raise ValueError("effect values must be finite numeric values")
    estimable_rows = effects["status"].isin(EFFECT_ESTIMABLE_STATUSES)
    if numeric_effect.loc[estimable_rows].isna().any():
        raise ValueError("estimable effect rows require finite effect values")
    effects = effects.copy(deep=True)
    effects["effect"] = numeric_effect

    variant_keys = (
        "method",
        "method_version",
        "resource",
        "resource_version",
        "score_semantics",
    )
    rows: list[dict[str, object]] = []
    comparison_groups = [
        "dataset",
        "analysis_track",
        "resource_mode",
        "contrast",
        "universe_id",
    ]
    for comparison_keys, group in effects.groupby(
        comparison_groups, observed=True, sort=False
    ):
        dataset, analysis_track, resource_mode, contrast, universe_id = comparison_keys
        variants = tuple(
            group.loc[:, variant_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for left_variant, right_variant in itertools.combinations(variants, 2):
            left_mask = pd.Series(True, index=group.index)
            right_mask = pd.Series(True, index=group.index)
            for column, value in zip(variant_keys, left_variant, strict=True):
                left_mask &= group[column].eq(value)
            for column, value in zip(variant_keys, right_variant, strict=True):
                right_mask &= group[column].eq(value)
            left_all = group.loc[left_mask]
            right_all = group.loc[right_mask]
            if (
                left_all.duplicated(list(EDGE_KEYS)).any()
                or right_all.duplicated(list(EDGE_KEYS)).any()
            ):
                raise ValueError("each method system must have unique LR edge effects")
            left_universe = set(
                left_all.loc[:, EDGE_KEYS].itertuples(index=False, name=None)
            )
            right_universe = set(
                right_all.loc[:, EDGE_KEYS].itertuples(index=False, name=None)
            )
            if left_universe != right_universe:
                raise ValueError(
                    "methods sharing universe_id must materialize the same "
                    "edge universe"
                )
            left = left_all.loc[left_all["status"].isin(EFFECT_ESTIMABLE_STATUSES)]
            right = right_all.loc[right_all["status"].isin(EFFECT_ESTIMABLE_STATUSES)]
            merged = left[[*EDGE_KEYS, "effect"]].merge(
                right[[*EDGE_KEYS, "effect"]],
                on=list(EDGE_KEYS),
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            merged = merged.loc[
                np.isfinite(pd.to_numeric(merged["effect_left"], errors="coerce"))
                & np.isfinite(pd.to_numeric(merged["effect_right"], errors="coerce"))
            ]
            enough = len(merged) >= minimum_shared_edges
            constant_rank = bool(
                enough
                and (
                    merged["effect_left"].nunique() < 2
                    or merged["effect_right"].nunique() < 2
                )
            )
            rank_estimable = enough and not constant_rank
            rho = (
                float(
                    spearmanr(merged["effect_left"], merged["effect_right"]).statistic
                )
                if rank_estimable
                else np.nan
            )
            nonzero = ~(merged["effect_left"].eq(0) & merged["effect_right"].eq(0))
            direction_edges = int(nonzero.sum())
            direction_agreement = (
                float(
                    np.mean(
                        np.sign(merged.loc[nonzero, "effect_left"])
                        == np.sign(merged.loc[nonzero, "effect_right"])
                    )
                )
                if direction_edges
                else np.nan
            )
            family_rho: list[float] = []
            for _, family in merged.groupby(
                list(FAMILY_KEYS), sort=False, observed=True
            ):
                if (
                    len(family) >= 3
                    and family["effect_left"].nunique() >= 2
                    and family["effect_right"].nunique() >= 2
                ):
                    value = spearmanr(
                        family["effect_left"], family["effect_right"]
                    ).statistic
                    if np.isfinite(value):
                        family_rho.append(float(value))
            reason = (
                None
                if rank_estimable
                else "constant_effect_rank"
                if constant_rank
                else "insufficient_shared_edge_universe"
            )
            row: dict[str, object] = {
                "dataset": dataset,
                "analysis_track": analysis_track,
                "resource_mode": resource_mode,
                "contrast": contrast,
                "universe_id": universe_id,
                "universe_edges": len(left_universe),
                "left_estimable_edges": len(left),
                "right_estimable_edges": len(right),
                "shared_edges": len(merged),
                "shared_edge_coverage": (
                    len(merged) / len(left_universe) if left_universe else np.nan
                ),
                "effect_spearman": rho,
                "macro_family_effect_spearman": (
                    float(np.mean(family_rho)) if family_rho else np.nan
                ),
                "n_estimable_families": len(family_rho),
                "direction_comparable_edges": direction_edges,
                "direction_agreement": direction_agreement,
                "status": "observed" if rank_estimable else "not_estimable",
                "reason_code": reason,
            }
            for side, variant in (("left", left_variant), ("right", right_variant)):
                for column, value in zip(variant_keys, variant, strict=True):
                    row[f"{column}_{side}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "COMPARISON_ELIGIBLE_STATUSES",
    "EDGE_KEYS",
    "EFFECT_ESTIMABLE_STATUSES",
    "FAMILY_KEYS",
    "METHOD_IDENTITY_KEYS",
    "MOLECULAR_LR_EQUIVALENCE_COLUMN",
    "PERFORMANCE_IDENTITY_KEYS",
    "REQUIRED_SCORE_COLUMNS",
    "RESOURCE_MODES",
    "SCORE_STATUSES",
    "SYNTHETIC_TRUTH_SCOPES",
    "aggregate_loso_primary_endpoint",
    "cross_method_concordance",
    "external_long_to_score_table",
    "paired_differential_loso_reproducibility",
    "paired_edge_effects",
    "score_coverage_summary",
    "summarize_loso_primary_endpoint",
    "summarize_run_performance",
    "synthetic_edge_truth_metrics",
    "unpaired_differential_split_half_reproducibility",
    "unpaired_edge_effects",
    "unpaired_leave_one_subject_influence",
    "validate_score_table",
    "within_context_reproducibility",
]
