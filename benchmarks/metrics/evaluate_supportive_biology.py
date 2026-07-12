"""Evaluate preregistered supportive biology without treating it as truth.

The locked observations are a silver-standard interpretation aid for real data.
They are not edge-level labels and must never be used to compute AUROC/AUPRC or
to tune a method.  This module keeps resource coverage, estimability, direction,
and rank support separate so that an absent result is never counted as a
biological contradiction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from benchmarks.metrics.multicondition import paired_edge_effects, unpaired_edge_effects

EVALUATION_PROTOCOL = "supportive-biology-evaluation-v1"
EVIDENCE_CLASSES = frozenset({"exact", "partial", "not_supported", "not_estimable"})
REPORT_SUPPORT_STATUSES = frozenset(
    {"supported", "partial", "discordant", "not_covered", "not_estimable"}
)
ESTIMABLE_STATUSES = frozenset(
    {"observed", "exploratory", "descriptive", "ok", "complete", "supported"}
)
RESOURCE_UNAVAILABLE_STATUSES = frozenset(
    {"resource_unavailable", "unsupported_resource", "not_covered"}
)
IDENTITY_COLUMNS = (
    "method",
    "method_version",
    "analysis_track",
    "resource",
    "resource_version",
    "resource_mode",
    "score_semantics",
    "universe_id",
)
EDGE_COLUMNS = ("sender", "receiver", "interaction_id", "ligand", "receptor")
EXTERNAL_LONG_EFFECT_COLUMNS = (
    "run_id",
    "dataset_id",
    "method_id",
    "method_version",
    "analysis_track",
    "resource_mode",
    "resource_id",
    "resource_version",
    "universe_id",
    "universe_member",
    "universe_size",
    "sample_id",
    "subject_id",
    "context_json",
    *EDGE_COLUMNS,
    "target",
    "score",
    "score_name",
    "score_direction",
    "rank",
    "status",
)
OUTPUT_COLUMNS = (
    "truth_set_id",
    "evaluation_protocol",
    "locked_at",
    "mapping_id",
    "dataset",
    "observation_id",
    "expected_direction",
    "expected_metric",
    *IDENTITY_COLUMNS,
    "evidence_class",
    "support_status",
    "observed_direction",
    "n_components_expected",
    "n_components_covered",
    "n_components_estimable",
    "n_components_strong",
    "n_components_directional",
    "n_components_opposite",
    "best_expected_direction_percentile",
    "best_effect",
    "median_effect",
    "effect_semantics",
    "top_fraction_threshold",
    "network_top_fraction_threshold",
    "component_evidence_json",
    "evidence_note",
    "status",
    "reason_code",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _mapping_id(mappings: Mapping[str, Mapping[str, str]]) -> str:
    digest = hashlib.sha256(_canonical_json(mappings).encode("utf-8")).hexdigest()
    return f"biology_mapping_{digest[:32]}"


def load_supportive_biology(path: str | Path) -> dict[str, Any]:
    """Load and validate the locked supportive-biology specification."""

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("supportive biology YAML must contain a mapping")
    truth = dict(cast(Mapping[str, Any], loaded))
    required = {"truth_set_id", "locked_at", "datasets", "evaluation_policy"}
    missing = required.difference(truth)
    if missing:
        raise ValueError(f"supportive biology YAML is missing: {sorted(missing)}")
    datasets = truth["datasets"]
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("supportive biology YAML must define datasets")
    policy = truth["evaluation_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("evaluation_policy must be a mapping")
    required_policy = {
        "preregistered_before_method_outputs": True,
        "real_data_is_not_edge_level_ground_truth": True,
        "do_not_compute_real_data_edge_auroc": True,
        "resource_absence_is_not_method_failure": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            raise ValueError(f"evaluation_policy must set {key}=true")
    for dataset, raw_spec in datasets.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"dataset {dataset!r} must contain a mapping")
        observations = raw_spec.get("expected_observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"dataset {dataset!r} has no expected_observations")
        identifiers: list[str] = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError("each expected observation must be a mapping")
            for field in ("id", "direction", "metric"):
                if not str(observation.get(field, "")).strip():
                    raise ValueError(f"dataset {dataset!r} observation lacks {field!r}")
            identifiers.append(str(observation["id"]))
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"dataset {dataset!r} has duplicate observation IDs")
    return truth


def load_biology_mappings(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Load explicit input aliases without changing locked observations."""

    if path is None:
        return {"cell_types": {}, "genes": {}}
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if loaded is None:
        return {"cell_types": {}, "genes": {}}
    if not isinstance(loaded, Mapping):
        raise ValueError("biology mapping config must contain a mapping")
    unknown = set(loaded).difference({"cell_types", "genes"})
    if unknown:
        raise ValueError(
            "biology mappings may contain only cell_types and genes; "
            f"unexpected={sorted(map(str, unknown))}"
        )
    result: dict[str, dict[str, str]] = {}
    for section in ("cell_types", "genes"):
        values = loaded.get(section, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"biology mapping section {section!r} must be a mapping")
        aliases = {str(key): str(value) for key, value in values.items()}
        if any(not key or not value for key, value in aliases.items()):
            raise ValueError(
                "biology mapping aliases and canonical values are nonempty"
            )
        result[section] = aliases
    return result


def _identity_value(table: pd.DataFrame, name: str, default: str) -> pd.Series:
    if name in table:
        return table[name].astype("string").fillna(default).astype(str)
    aliases = {
        "method": "method_id",
        "resource": "resource_id",
        "score_semantics": "score_name",
    }
    alias = aliases.get(name)
    if alias is not None and alias in table:
        return table[alias].astype("string").fillna(default).astype(str)
    return pd.Series(default, index=table.index, dtype=str)


def _normalize_effect_table(
    table: pd.DataFrame,
    *,
    mappings: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    required = {"sender", "receiver", "ligand", "receptor", "effect", "status"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"biology effect table is missing columns: {sorted(missing)}")
    result = table.copy(deep=True)
    defaults = {
        "method": "unknown",
        "method_version": "unknown",
        "analysis_track": "lr_stlr",
        "resource": "unknown",
        "resource_version": "unknown",
        "resource_mode": "unknown",
        "score_semantics": "differential_rank_strength",
        "universe_id": "unknown",
    }
    for name, default in defaults.items():
        result[name] = _identity_value(result, name, default)
    if "interaction_id" not in result:
        result["interaction_id"] = [
            f"{ligand}|{receptor}"
            for ligand, receptor in zip(
                result["ligand"].astype(str),
                result["receptor"].astype(str),
                strict=True,
            )
        ]
    if "target" not in result:
        result["target"] = ""
    if "pathway" not in result:
        result["pathway"] = ""
    cell_aliases = mappings.get("cell_types", {})
    gene_aliases = mappings.get("genes", {})
    for field in ("sender", "receiver"):
        result[field] = result[field].astype(str).replace(cell_aliases)
    for field in ("ligand", "receptor", "target"):
        result[field] = result[field].astype(str).replace(gene_aliases)
    result["interaction_id"] = result["interaction_id"].astype(str)
    result["pathway"] = result["pathway"].astype(str)
    result["status"] = result["status"].astype(str)
    effect = pd.to_numeric(result["effect"], errors="coerce")
    supplied = result["effect"].notna()
    if ((supplied & effect.isna()) | np.isinf(effect.fillna(0.0))).any():
        raise ValueError("biology effect values must be finite or missing")
    result["effect"] = effect
    estimable = result["status"].isin(ESTIMABLE_STATUSES)
    if result.loc[estimable, "effect"].isna().any():
        raise ValueError("estimable biology effect rows require a finite effect")
    key = [*IDENTITY_COLUMNS, *EDGE_COLUMNS, "target", "pathway"]
    if result.duplicated(key).any():
        raise ValueError("biology effect table has duplicate method-edge rows")
    return result


def effect_table_from_score_table(
    table: pd.DataFrame,
    *,
    reference: str,
    target: str,
    design: str,
    min_support: int = 3,
    contrast: str | None = None,
) -> pd.DataFrame:
    """Compute exploratory edge effects from a validated LR score table."""

    if design == "paired":
        return cast(
            pd.DataFrame,
            paired_edge_effects(
                table,
                reference=reference,
                target=target,
                min_pairs=min_support,
                contrast=contrast,
            ),
        )
    if design == "independent":
        return cast(
            pd.DataFrame,
            unpaired_edge_effects(
                table,
                reference=reference,
                target=target,
                min_subjects=min_support,
                contrast=contrast,
            ),
        )
    raise ValueError("design must be 'paired' or 'independent'")


def _validated_frozen_edge_ids(
    working: pd.DataFrame,
    *,
    edge_columns: Sequence[str],
    sample_codes: np.ndarray[Any],
    sample_count: int,
    declared_size: int,
) -> tuple[np.ndarray[Any], pd.DataFrame]:
    """Validate a materialized frozen universe and return compact edge IDs."""

    changes: np.ndarray[Any] = np.empty(len(sample_codes), dtype=bool)
    changes[0] = True
    changes[1:] = sample_codes[1:] != sample_codes[:-1]
    starts = np.flatnonzero(changes)
    ends = np.append(starts[1:], len(sample_codes))
    block_codes = sample_codes[starts]
    ordered_blocks = (
        len(starts) == sample_count
        and np.unique(block_codes).size == sample_count
        and np.all((ends - starts) == declared_size)
    )
    if ordered_blocks:
        first = working.iloc[starts[0] : ends[0]].loc[:, edge_columns]
        first = first.reset_index(drop=True)
        exact_match = True
        for start, end in zip(starts[1:], ends[1:], strict=True):
            candidate = working.iloc[start:end].loc[:, edge_columns]
            candidate = candidate.reset_index(drop=True)
            if any(
                not candidate[column].equals(first[column]) for column in edge_columns
            ):
                exact_match = False
                break
        if exact_match:
            edge_ids: np.ndarray[Any] = np.empty(len(working), dtype=np.uint64)
            dense_ids: np.ndarray[Any] = np.arange(declared_size, dtype=np.uint64)
            for start, end in zip(starts, ends, strict=True):
                edge_ids[start:end] = dense_ids
            universe = first.copy()
            universe.insert(0, "_edge_id", dense_ids)
            return edge_ids, universe

    edge_ids = pd.util.hash_pandas_object(
        working.loc[:, edge_columns], index=False, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)
    for sample_code in range(sample_count):
        sample_edge_ids = edge_ids[sample_codes == sample_code]
        if sample_edge_ids.size != declared_size:
            raise ValueError(
                "every sample must materialize the declared frozen universe"
            )
        if pd.unique(sample_edge_ids).size != declared_size:
            raise ValueError("external long table has duplicate sample-edge rows")
    first_edge = ~pd.Series(edge_ids, copy=False).duplicated().to_numpy()
    if int(first_edge.sum()) != declared_size:
        raise ValueError("declared universe_size disagrees with unique adapter edges")
    universe = working.loc[first_edge, edge_columns].copy()
    universe.insert(0, "_edge_id", edge_ids[first_edge])
    return edge_ids, universe


def effect_table_from_external_long(
    table: pd.DataFrame,
    *,
    context_key: str,
    reference: str,
    target: str,
    design: str,
    min_support: int = 3,
    contrast: str | None = None,
) -> pd.DataFrame:
    """Convert a fixed-universe adapter table and compute edge effects.

    Track B ligand-target rows use the same sample-level rank transformation as
    LR rows, but their original ``analysis_track`` is restored before biology
    evaluation.  This does not turn them into LR edges or make a sender claim;
    the source-agnostic sender and target-program receptor placeholders remain
    explicit in the effect table.
    """

    del contrast  # The external table stores sample contexts, not contrast labels.
    if design not in {"paired", "independent"}:
        raise ValueError("design must be 'paired' or 'independent'")
    if isinstance(min_support, bool) or not isinstance(min_support, int):
        raise ValueError("min_support must be an integer")
    if min_support < 2:
        raise ValueError("min_support must be at least 2")
    required = {
        "run_id",
        "dataset_id",
        "method_id",
        "method_version",
        "analysis_track",
        "resource_mode",
        "resource_id",
        "resource_version",
        "universe_id",
        "universe_member",
        "universe_size",
        "sample_id",
        "subject_id",
        "context_json",
        *EDGE_COLUMNS,
        "score",
        "score_name",
        "score_direction",
        "status",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"external long table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("external long table is empty")
    if table["run_id"].astype(str).nunique() != 1:
        raise ValueError(
            "select exactly one adapter run/CRYCHIC score view before evaluation"
        )
    constant_fields = {
        "dataset_id",
        "method_id",
        "method_version",
        "analysis_track",
        "resource_mode",
        "resource_id",
        "resource_version",
        "universe_id",
        "universe_size",
        "score_name",
        "score_direction",
    }
    for field in constant_fields:
        if table[field].astype(str).nunique() != 1:
            raise ValueError(f"external long field {field!r} must be constant per run")
    if not table["universe_member"].astype(bool).all():
        raise ValueError("biology evaluation accepts frozen-universe members only")

    working_columns = [
        "sample_id",
        "subject_id",
        "context_json",
        *EDGE_COLUMNS,
        "score",
        "status",
    ]
    for optional in ("target", "pathway", "rank"):
        if optional in table:
            working_columns.append(optional)
    working = table.loc[:, working_columns].copy()
    if "target" not in working:
        working["target"] = ""
    if "pathway" not in working:
        working["pathway"] = ""
    for field in ("sample_id", "subject_id", *EDGE_COLUMNS, "target", "pathway"):
        working[field] = working[field].astype("string").fillna("").astype(str)
    if working[["sample_id", "subject_id", *EDGE_COLUMNS]].eq("").any().any():
        raise ValueError("external long identifiers must be nonempty")
    context_mapping: dict[str, str] = {}
    for raw in working["context_json"].astype(str).drop_duplicates():
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or context_key not in parsed:
            raise ValueError(
                f"context_json must encode an object containing {context_key!r}"
            )
        context_mapping[raw] = str(parsed[context_key])
    working["context"] = working["context_json"].astype(str).map(context_mapping)
    available_contexts = set(working["context"])
    missing_contexts = {reference, target}.difference(available_contexts)
    if missing_contexts:
        raise ValueError(f"external long table is missing contexts: {missing_contexts}")

    edge_columns = [*EDGE_COLUMNS, "target", "pathway"]
    declared_size = int(str(table["universe_size"].iloc[0]))
    sample_codes, sample_labels = pd.factorize(working["sample_id"], sort=False)
    if (sample_codes < 0).any():
        raise ValueError("external long sample_id values must be nonempty")

    # Validate the materialized frozen universe once, then aggregate on compact
    # numeric keys instead of repeatedly grouping seven large string columns.
    edge_ids, universe = _validated_frozen_edge_ids(
        working,
        edge_columns=edge_columns,
        sample_codes=sample_codes,
        sample_count=len(sample_labels),
        declared_size=declared_size,
    )

    subject_codes, _ = pd.factorize(working["subject_id"], sort=False)
    if (subject_codes < 0).any():
        raise ValueError("external long subject_id values must be nonempty")
    context_values = working["context"].to_numpy(dtype=str, copy=False)
    context_codes = np.select(
        [context_values == reference, context_values == target],
        [0, 1],
        default=-1,
    ).astype(np.int8, copy=False)

    status_map = {
        "ok": "observed",
        "not_returned": "not_predicted",
        "resource_unavailable": "resource_unavailable",
        "unsupported_resource": "not_supported",
        "insufficient_cells": "cell_type_missing",
        "method_failed": "failed",
        "missing": "missing",
    }
    metric_status = working["status"].astype(str).replace(status_map)
    allowed_statuses = {
        "observed",
        "not_predicted",
        "resource_unavailable",
        "not_supported",
        "cell_type_missing",
        "failed",
        "missing",
    }
    invalid_statuses = set(metric_status).difference(allowed_statuses)
    if invalid_statuses:
        raise ValueError(
            f"external long table has invalid statuses: {invalid_statuses}"
        )
    score = pd.to_numeric(working["score"], errors="coerce")
    observed = metric_status.eq("observed")
    if score.loc[observed].isna().any() or np.isinf(score.fillna(0.0)).any():
        raise ValueError("observed external rows require finite scores")
    comparison_strength: np.ndarray[Any] = np.full(len(working), np.nan, dtype=float)
    direction = str(table["score_direction"].iloc[0])
    if direction not in {"higher", "lower"}:
        raise ValueError("score_direction must be 'higher' or 'lower'")
    if "rank" in working:
        supplied_rank = pd.to_numeric(working["rank"], errors="coerce")
        if supplied_rank.loc[observed].isna().any():
            raise ValueError("observed external rows require a finite adapter rank")
        observed_rank = supplied_rank.loc[observed]
    else:
        oriented = score if direction == "higher" else -score
        observed_rank = (
            oriented.loc[observed]
            .groupby(working.loc[observed, "sample_id"], observed=True, sort=False)
            .rank(ascending=False, method="average")
        )
    comparable = metric_status.isin({"observed", "not_predicted"}).to_numpy()
    comparable_size = np.bincount(
        sample_codes,
        weights=comparable.astype(float),
        minlength=len(sample_labels),
    )
    observed_mask = observed.to_numpy()
    observed_rank_array = observed_rank.to_numpy(dtype=float, copy=False)
    comparison_strength[observed_mask] = (
        1.0 - (observed_rank_array - 1.0) / comparable_size[sample_codes[observed_mask]]
    )
    comparison_strength[metric_status.eq("not_predicted").to_numpy()] = 0.0

    exceptional_statuses = {"resource_unavailable", "not_supported", "failed"}
    if set(metric_status).isdisjoint(exceptional_statuses):
        edge_state = universe.loc[:, ["_edge_id"]].copy()
        edge_state["all_resource_unavailable"] = False
        edge_state["all_not_supported"] = False
        edge_state["any_failed"] = False
    else:
        state_rows = pd.DataFrame(
            {
                "_edge_id": edge_ids,
                "_resource_unavailable": metric_status.eq(
                    "resource_unavailable"
                ).to_numpy(),
                "_not_supported": metric_status.eq("not_supported").to_numpy(),
                "_failed": metric_status.eq("failed").to_numpy(),
            }
        )
        edge_state = (
            state_rows.groupby("_edge_id", observed=True, sort=False, dropna=False)
            .agg(
                all_resource_unavailable=("_resource_unavailable", "min"),
                all_not_supported=("_not_supported", "min"),
                any_failed=("_failed", "max"),
            )
            .reset_index()
        )
    usable_mask = np.isfinite(comparison_strength) & (context_codes >= 0)
    usable = pd.DataFrame(
        {
            "_edge_id": edge_ids[usable_mask],
            "_subject_code": subject_codes[usable_mask],
            "_context_code": context_codes[usable_mask],
            "comparison_strength": comparison_strength[usable_mask],
        }
    )
    subject_context = (
        usable.groupby(
            ["_edge_id", "_subject_code", "_context_code"],
            observed=True,
            sort=False,
            dropna=False,
        )["comparison_strength"]
        .mean()
        .reset_index()
    )
    if design == "paired":
        left = subject_context.loc[
            subject_context["_context_code"].eq(0),
            ["_edge_id", "_subject_code", "comparison_strength"],
        ].rename(columns={"comparison_strength": "reference_strength"})
        right = subject_context.loc[
            subject_context["_context_code"].eq(1),
            ["_edge_id", "_subject_code", "comparison_strength"],
        ].rename(columns={"comparison_strength": "target_strength"})
        paired = left.merge(
            right,
            on=["_edge_id", "_subject_code"],
            how="inner",
            validate="one_to_one",
        )
        paired["_difference"] = paired["target_strength"] - paired["reference_strength"]
        summary = (
            paired.groupby("_edge_id", observed=True, sort=False, dropna=False)[
                "_difference"
            ]
            .agg(effect="mean", n_pairs="size")
            .reset_index()
        )
        support = summary["n_pairs"].ge(min_support)
        count_columns = {"n_pairs": summary["n_pairs"]}
        effect_semantics = "target_minus_reference_paired_comparison_strength"
    else:
        overlap = set(subject_codes[context_codes == 0]) & set(
            subject_codes[context_codes == 1]
        )
        if overlap:
            overlap_labels = sorted(
                set(working.loc[np.isin(subject_codes, list(overlap)), "subject_id"])
            )
            raise ValueError(
                "independent design contains subjects in both contexts: "
                f"{overlap_labels}"
            )
        context_summary = (
            subject_context.groupby(
                ["_edge_id", "_context_code"],
                observed=True,
                sort=False,
                dropna=False,
            )["comparison_strength"]
            .agg(subject_mean="mean", n_subjects="size")
            .reset_index()
        )
        reference_summary = context_summary.loc[
            context_summary["_context_code"].eq(0),
            ["_edge_id", "subject_mean", "n_subjects"],
        ].rename(
            columns={
                "subject_mean": "reference_mean",
                "n_subjects": "n_reference_subjects",
            }
        )
        target_summary = context_summary.loc[
            context_summary["_context_code"].eq(1),
            ["_edge_id", "subject_mean", "n_subjects"],
        ].rename(
            columns={
                "subject_mean": "target_mean",
                "n_subjects": "n_target_subjects",
            }
        )
        summary = reference_summary.merge(
            target_summary,
            on="_edge_id",
            how="outer",
            validate="one_to_one",
        )
        summary["effect"] = summary["target_mean"] - summary["reference_mean"]
        support = summary["n_reference_subjects"].fillna(0).ge(min_support) & summary[
            "n_target_subjects"
        ].fillna(0).ge(min_support)
        count_columns = {
            "n_reference_subjects": summary["n_reference_subjects"],
            "n_target_subjects": summary["n_target_subjects"],
        }
        effect_semantics = (
            "target_minus_reference_equal_subject_mean_comparison_strength"
        )
    summary["_support"] = support
    for name, values in count_columns.items():
        summary[name] = values
    effects = universe.merge(summary, on="_edge_id", how="left", validate="one_to_one")
    effects = effects.merge(
        edge_state, on="_edge_id", how="left", validate="one_to_one"
    )
    effects["status"] = np.select(
        [
            effects["_support"].fillna(False).to_numpy(dtype=bool),
            effects["all_resource_unavailable"].fillna(False).to_numpy(dtype=bool),
            effects["all_not_supported"].fillna(False).to_numpy(dtype=bool),
            effects["any_failed"].fillna(False).to_numpy(dtype=bool),
        ],
        ["exploratory", "resource_unavailable", "not_supported", "failed"],
        default="not_estimable",
    )
    effects.loc[~effects["status"].eq("exploratory"), "effect"] = np.nan
    identity = {
        "dataset": str(table["dataset_id"].iloc[0]),
        "method": str(table["method_id"].iloc[0]),
        "method_version": str(table["method_version"].iloc[0]),
        "analysis_track": str(table["analysis_track"].iloc[0]),
        "resource": str(table["resource_id"].iloc[0]),
        "resource_version": str(table["resource_version"].iloc[0]),
        "resource_mode": str(table["resource_mode"].iloc[0]),
        "score_semantics": str(table["score_name"].iloc[0]),
        "universe_id": str(table["universe_id"].iloc[0]),
    }
    for field, value in identity.items():
        effects[field] = value
    effects = effects.drop(columns="_edge_id")
    effects["effect_semantics"] = effect_semantics
    effects["reference"] = reference
    effects["target_context"] = target
    effects["reason_code"] = np.select(
        [
            effects["status"].eq("resource_unavailable").to_numpy(dtype=bool),
            effects["status"].eq("not_supported").to_numpy(dtype=bool),
            effects["status"].eq("failed").to_numpy(dtype=bool),
            effects["status"].eq("not_estimable").to_numpy(dtype=bool),
        ],
        [
            "resource_unavailable",
            "method_or_resource_not_supported",
            "method_run_failed",
            "insufficient_subject_support",
        ],
        default=None,
    )
    return effects.drop(
        columns=[
            "_support",
            "all_resource_unavailable",
            "all_not_supported",
            "any_failed",
        ],
        errors="ignore",
    )


def _tokens(value: object) -> frozenset[str]:
    return frozenset(
        token.strip().upper()
        for token in str(value).replace("+", "&").replace("|", "&").split("&")
        if token.strip()
    )


def _gene_match(values: pd.Series, expected: str) -> pd.Series:
    symbol = expected.upper()
    return values.map(lambda value: symbol in _tokens(value))


def _cell_match(values: pd.Series, expected: str) -> pd.Series:
    canonical = expected.casefold()
    return values.astype(str).str.casefold().eq(canonical)


def _expected_sign(direction: str, *, reference: str, target: str) -> float:
    if not direction.endswith("_up"):
        raise ValueError(f"unsupported expected direction: {direction!r}")
    label = direction[: -len("_up")]
    normalized = label.casefold().replace("-", "_").replace(" ", "_")
    reference_label = reference.casefold().replace("-", "_").replace(" ", "_")
    target_label = target.casefold().replace("-", "_").replace(" ", "_")
    if normalized == target_label:
        return 1.0
    if normalized == reference_label:
        return -1.0
    raise ValueError(
        f"locked direction {direction!r} does not match {target!r} or {reference!r}"
    )


def _rank_percentiles(table: pd.DataFrame, expected_sign: float) -> pd.Series:
    estimable = table["status"].isin(ESTIMABLE_STATUSES) & table["effect"].notna()
    result = pd.Series(np.nan, index=table.index, dtype=float)
    selected = table.loc[estimable].copy()
    if selected.empty:
        return result
    selected["_signed"] = expected_sign * selected["effect"]
    grouping = ["sender", "receiver"]
    ranks = selected.groupby(grouping, observed=True, sort=False)["_signed"].rank(
        ascending=False, method="average"
    )
    sizes = selected.groupby(grouping, observed=True, sort=False)["_signed"].transform(
        "size"
    )
    percentiles = np.where(sizes.eq(1), 1.0, 1.0 - (ranks - 1.0) / (sizes - 1.0))
    result.loc[selected.index] = percentiles
    return result


def _component_result(
    *,
    component_id: str,
    kind: str,
    covered: bool,
    estimable: bool,
    strength: str,
    reason_code: str,
    effect: float | None = None,
    percentile: float | None = None,
    n_candidates: int = 0,
) -> dict[str, object]:
    return {
        "component_id": component_id,
        "kind": kind,
        "covered": covered,
        "estimable": estimable,
        "strength": strength,
        "reason_code": reason_code,
        "effect": effect,
        "expected_direction_percentile": percentile,
        "n_candidates": n_candidates,
    }


def _evaluate_selection(
    table: pd.DataFrame,
    mask: pd.Series,
    *,
    component_id: str,
    kind: str,
    expected_sign: float,
    percentiles: pd.Series,
    top_fraction: float,
    resource_mask: pd.Series | None = None,
) -> dict[str, object]:
    selected = table.loc[mask]
    if selected.empty:
        if resource_mask is not None and bool(resource_mask.any()):
            return _component_result(
                component_id=component_id,
                kind=kind,
                covered=True,
                estimable=False,
                strength="not_estimable",
                reason_code="locked_component_cell_pair_not_estimable",
            )
        return _component_result(
            component_id=component_id,
            kind=kind,
            covered=False,
            estimable=False,
            strength="not_covered",
            reason_code="locked_component_absent_from_frozen_resource",
        )
    if selected["status"].isin(RESOURCE_UNAVAILABLE_STATUSES).all():
        return _component_result(
            component_id=component_id,
            kind=kind,
            covered=False,
            estimable=False,
            strength="not_covered",
            reason_code="locked_component_resource_unavailable",
            n_candidates=len(selected),
        )
    usable = selected.loc[
        selected["status"].isin(ESTIMABLE_STATUSES) & selected["effect"].notna()
    ]
    if usable.empty:
        reasons = sorted(set(selected["status"].astype(str)))
        return _component_result(
            component_id=component_id,
            kind=kind,
            covered=True,
            estimable=False,
            strength="not_estimable",
            reason_code="locked_component_not_estimable:" + ",".join(reasons),
            n_candidates=len(selected),
        )
    signed = expected_sign * usable["effect"]
    best_index = signed.idxmax()
    best_signed = float(signed.loc[best_index])
    best_effect = float(cast(float, usable.loc[best_index, "effect"]))
    percentile = float(percentiles.loc[best_index])
    if best_signed > 0 and percentile >= 1.0 - top_fraction:
        strength = "strong"
        reason = "expected_direction_and_top_rank"
    elif best_signed > 0:
        strength = "directional"
        reason = "expected_direction_below_top_rank_threshold"
    else:
        strength = "opposite"
        reason = "no_candidate_in_expected_direction"
    return _component_result(
        component_id=component_id,
        kind=kind,
        covered=True,
        estimable=True,
        strength=strength,
        reason_code=reason,
        effect=best_effect,
        percentile=percentile,
        n_candidates=len(selected),
    )


def _network_family_scores(
    table: pd.DataFrame,
    *,
    expected_sign: float,
    tail_fraction: float,
) -> pd.DataFrame:
    usable = table.loc[
        table["status"].isin(ESTIMABLE_STATUSES) & table["effect"].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame(
            columns=["sender", "receiver", "effect", "percentile", "n_edges"]
        )
    usable["_signed"] = expected_sign * usable["effect"]
    rows: list[dict[str, object]] = []
    for (sender, receiver), family in usable.groupby(
        ["sender", "receiver"], observed=True, sort=False
    ):
        n_tail = max(1, math.ceil(len(family) * tail_fraction))
        signed = family["_signed"].nlargest(n_tail)
        representative = signed.index[0]
        rows.append(
            {
                "sender": str(sender),
                "receiver": str(receiver),
                "signed_score": float(signed.mean()),
                "effect": float(family.loc[representative, "effect"]),
                "n_edges": len(family),
            }
        )
    result = pd.DataFrame(rows)
    ranks = result["signed_score"].rank(ascending=False, method="average")
    result["percentile"] = np.where(
        len(result) == 1,
        1.0,
        1.0 - (ranks - 1.0) / (len(result) - 1.0),
    )
    return result


def _evaluate_network_pair(
    table: pd.DataFrame,
    family_scores: pd.DataFrame,
    *,
    sender: str,
    receiver: str,
    component_id: str,
    network_top_fraction: float,
) -> dict[str, object]:
    cell_types = set(table["sender"]) | set(table["receiver"])
    if sender not in cell_types or receiver not in cell_types:
        return _component_result(
            component_id=component_id,
            kind="network_pair",
            covered=True,
            estimable=False,
            strength="not_estimable",
            reason_code="locked_cell_type_absent_from_benchmark_ontology",
        )
    selected = family_scores.loc[
        _cell_match(family_scores["sender"], sender)
        & _cell_match(family_scores["receiver"], receiver)
    ]
    if selected.empty:
        return _component_result(
            component_id=component_id,
            kind="network_pair",
            covered=True,
            estimable=False,
            strength="not_estimable",
            reason_code="network_family_has_no_estimable_edges",
        )
    row = selected.sort_values("signed_score", ascending=False).iloc[0]
    score = float(row["signed_score"])
    percentile = float(row["percentile"])
    if score > 0 and percentile >= 1.0 - network_top_fraction:
        strength = "strong"
        reason = "network_family_expected_direction_and_top_rank"
    elif score > 0:
        strength = "directional"
        reason = "network_family_expected_direction_below_rank_threshold"
    else:
        strength = "opposite"
        reason = "network_family_not_in_expected_direction"
    return _component_result(
        component_id=component_id,
        kind="network_pair",
        covered=True,
        estimable=True,
        strength=strength,
        reason_code=reason,
        effect=float(row["effect"]),
        percentile=percentile,
        n_candidates=int(row["n_edges"]),
    )


def _network_components(
    table: pd.DataFrame,
    observation: Mapping[str, Any],
    *,
    expected_sign: float,
    network_top_fraction: float,
) -> list[dict[str, object]]:
    families = _network_family_scores(
        table,
        expected_sign=expected_sign,
        tail_fraction=0.05,
    )
    sender_values = observation.get("senders")
    if sender_values is None and observation.get("sender") is not None:
        sender_values = [observation["sender"]]
    receiver_values = observation.get("receivers")
    if receiver_values is None and observation.get("receiver") is not None:
        receiver_values = [observation["receiver"]]
    if sender_values is not None and receiver_values is not None:
        return [
            _evaluate_network_pair(
                table,
                families,
                sender=str(sender),
                receiver=str(receiver),
                component_id=f"network:{sender}->{receiver}",
                network_top_fraction=network_top_fraction,
            )
            for sender in cast(Sequence[object], sender_values)
            for receiver in cast(Sequence[object], receiver_values)
        ]
    cell_types = observation.get("cell_types")
    if isinstance(cell_types, Sequence) and not isinstance(cell_types, str):
        components: list[dict[str, object]] = []
        for cell_type in cell_types:
            label = str(cell_type)
            incident = families.loc[
                _cell_match(families["sender"], label)
                | _cell_match(families["receiver"], label)
            ]
            if incident.empty:
                components.append(
                    _component_result(
                        component_id=f"network_cell_type:{label}",
                        kind="network_cell_type",
                        covered=True,
                        estimable=False,
                        strength="not_estimable",
                        reason_code="cell_type_network_has_no_estimable_edges",
                    )
                )
                continue
            best = incident.sort_values("signed_score", ascending=False).iloc[0]
            score = float(best["signed_score"])
            percentile = float(best["percentile"])
            strong = score > 0 and percentile >= 1.0 - network_top_fraction
            components.append(
                _component_result(
                    component_id=f"network_cell_type:{label}",
                    kind="network_cell_type",
                    covered=True,
                    estimable=True,
                    strength="strong"
                    if strong
                    else "directional"
                    if score > 0
                    else "opposite",
                    reason_code=(
                        "cell_type_network_expected_direction_and_top_rank"
                        if strong
                        else "cell_type_network_expected_direction_below_rank_threshold"
                        if score > 0
                        else "cell_type_network_not_in_expected_direction"
                    ),
                    effect=float(best["effect"]),
                    percentile=percentile,
                    n_candidates=int(incident["n_edges"].sum()),
                )
            )
        return components
    return []


def _lr_components(
    table: pd.DataFrame,
    observation: Mapping[str, Any],
    *,
    expected_sign: float,
    percentiles: pd.Series,
    top_fraction: float,
) -> list[dict[str, object]]:
    interactions = observation.get("interactions")
    if not isinstance(interactions, list):
        return []
    target_track = (
        table["analysis_track"]
        .astype(str)
        .str.contains("ligand_target|target_program", case=False, regex=True)
        .any()
    )
    if target_track:
        return [
            _component_result(
                component_id=f"lr:{pair[0]!s}->{pair[1]!s}",
                kind="reported_lr",
                covered=True,
                estimable=False,
                strength="not_estimable",
                reason_code="track_b_does_not_estimate_lr_or_sender_identity",
            )
            for pair in interactions
            if isinstance(pair, Sequence)
            and not isinstance(pair, str)
            and len(pair) == 2
        ]
    senders = observation.get("senders")
    if senders is None and observation.get("sender") is not None:
        senders = [observation["sender"]]
    receivers = observation.get("receivers")
    if receivers is None and observation.get("receiver") is not None:
        receivers = [observation["receiver"]]
    sender_values = [None] if senders is None else list(cast(Sequence[object], senders))
    receiver_values = (
        [None] if receivers is None else list(cast(Sequence[object], receivers))
    )
    components: list[dict[str, object]] = []
    for pair in interactions:
        if not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2:
            raise ValueError("locked interactions must contain [ligand, receptor]")
        ligand, receptor = map(str, pair)
        for sender in sender_values:
            for receiver in receiver_values:
                resource_mask = _gene_match(table["ligand"], ligand) & _gene_match(
                    table["receptor"], receptor
                )
                mask = resource_mask.copy()
                labels = []
                if sender is not None:
                    mask &= _cell_match(table["sender"], str(sender))
                    labels.append(str(sender))
                if receiver is not None:
                    mask &= _cell_match(table["receiver"], str(receiver))
                    labels.append(str(receiver))
                location = "->".join(labels) if labels else "any_cell_pair"
                components.append(
                    _evaluate_selection(
                        table,
                        mask,
                        component_id=f"lr:{location}:{ligand}->{receptor}",
                        kind="reported_lr",
                        expected_sign=expected_sign,
                        percentiles=percentiles,
                        top_fraction=top_fraction,
                        resource_mask=resource_mask,
                    )
                )
    return components


def _program_components(
    table: pd.DataFrame,
    observation: Mapping[str, Any],
    *,
    expected_sign: float,
    percentiles: pd.Series,
    top_fraction: float,
) -> list[dict[str, object]]:
    ligands = observation.get("ligands")
    if not isinstance(ligands, list):
        return []
    receiver = observation.get("receiver")
    senders = observation.get("senders", [])
    receptors = [
        str(value)
        for value in observation.get("receptors_or_programs", [])
        if str(value).upper() != "EMT"
    ]
    target_track = (
        table["analysis_track"]
        .astype(str)
        .str.contains("ligand_target|target_program", case=False, regex=True)
        .any()
    )
    components: list[dict[str, object]] = []
    for ligand_value in ligands:
        ligand = str(ligand_value)
        resource_mask = _gene_match(table["ligand"], ligand)
        if not target_track and receptors:
            receptor_mask = pd.Series(False, index=table.index)
            for receptor in receptors:
                receptor_mask |= _gene_match(table["receptor"], receptor)
            resource_mask &= receptor_mask
        mask = resource_mask.copy()
        if receiver is not None:
            mask &= _cell_match(table["receiver"], str(receiver))
        if not target_track and senders:
            sender_mask = pd.Series(False, index=table.index)
            for sender in cast(Sequence[object], senders):
                sender_mask |= _cell_match(table["sender"], str(sender))
            mask &= sender_mask
        components.append(
            _evaluate_selection(
                table,
                mask,
                component_id=f"program_ligand:{ligand}",
                kind=("ligand_target_program" if target_track else "lr_program_proxy"),
                expected_sign=expected_sign,
                percentiles=percentiles,
                top_fraction=top_fraction,
                resource_mask=resource_mask,
            )
        )
    return components


def _pathway_components(
    table: pd.DataFrame,
    observation: Mapping[str, Any],
    *,
    expected_sign: float,
    percentiles: pd.Series,
    top_fraction: float,
) -> list[dict[str, object]]:
    pathways = observation.get("pathways")
    if not isinstance(pathways, list):
        return []
    receiver = observation.get("receiver")
    components: list[dict[str, object]] = []
    has_annotation = table["pathway"].astype(str).str.strip().ne("").any()
    for pathway_value in pathways:
        pathway = str(pathway_value)
        if not has_annotation:
            components.append(
                _component_result(
                    component_id=f"pathway:{pathway}",
                    kind="pathway_program",
                    covered=True,
                    estimable=False,
                    strength="not_estimable",
                    reason_code="pathway_annotation_unavailable",
                )
            )
            continue
        mask = table["pathway"].str.casefold().eq(pathway.casefold())
        if receiver is not None:
            mask &= _cell_match(table["receiver"], str(receiver))
        components.append(
            _evaluate_selection(
                table,
                mask,
                component_id=f"pathway:{pathway}",
                kind="pathway_program",
                expected_sign=expected_sign,
                percentiles=percentiles,
                top_fraction=top_fraction,
            )
        )
    return components


def _json_components(components: Sequence[Mapping[str, object]]) -> str:
    cleaned: list[dict[str, object]] = []
    for component in components:
        row: dict[str, object] = {}
        for key, value in component.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
            else:
                row[key] = value
        cleaned.append(row)
    return _canonical_json(cleaned)


def _summarize_components(
    components: Sequence[Mapping[str, object]],
    *,
    expected_direction: str,
    reference: str,
    target: str,
) -> dict[str, object]:
    strengths = [str(component["strength"]) for component in components]
    covered = sum(bool(component["covered"]) for component in components)
    estimable = sum(bool(component["estimable"]) for component in components)
    strong = strengths.count("strong")
    directional = strengths.count("directional")
    opposite = strengths.count("opposite")
    if components and strong == len(components):
        evidence_class = "exact"
        support_status = "supported"
        status = "observed"
        reason = "all_locked_components_top_ranked_in_expected_direction"
    elif strong + directional > 0:
        evidence_class = "partial"
        support_status = "partial"
        status = "observed"
        reason = "subset_or_directional_support_for_locked_components"
    elif estimable > 0:
        evidence_class = "not_supported"
        support_status = "discordant"
        status = "observed"
        reason = "estimable_components_do_not_support_expected_direction"
    else:
        evidence_class = "not_estimable"
        all_uncovered = bool(components) and all(
            strength == "not_covered" for strength in strengths
        )
        support_status = "not_covered" if all_uncovered else "not_estimable"
        status = "not_estimable"
        reason = (
            "locked_components_absent_from_frozen_resource"
            if all_uncovered
            else "locked_components_not_estimable"
        )
    effects = [
        float(cast(float, component["effect"]))
        for component in components
        if component.get("effect") is not None
    ]
    percentiles = [
        float(cast(float, component["expected_direction_percentile"]))
        for component in components
        if component.get("expected_direction_percentile") is not None
    ]
    expected_sign = _expected_sign(
        expected_direction, reference=reference, target=target
    )
    signed = np.asarray(effects, dtype=float) * expected_sign
    if not len(signed):
        observed_direction = ""
    elif np.all(signed > 0):
        observed_direction = expected_direction
    elif np.all(signed <= 0):
        opposite_label = reference if expected_sign > 0 else target
        observed_direction = f"{opposite_label}_up"
    else:
        observed_direction = "mixed"
    return {
        "evidence_class": evidence_class,
        "support_status": support_status,
        "observed_direction": observed_direction,
        "n_components_expected": len(components),
        "n_components_covered": covered,
        "n_components_estimable": estimable,
        "n_components_strong": strong,
        "n_components_directional": directional,
        "n_components_opposite": opposite,
        "best_expected_direction_percentile": (
            max(percentiles) if percentiles else math.nan
        ),
        "best_effect": (effects[int(np.argmax(signed))] if len(signed) else math.nan),
        "median_effect": float(np.median(effects)) if effects else math.nan,
        "status": status,
        "reason_code": reason,
    }


def evaluate_supportive_biology(
    effect_table: pd.DataFrame,
    truth: Mapping[str, Any] | str | Path,
    *,
    dataset_id: str,
    reference: str,
    target: str,
    mappings: Mapping[str, Mapping[str, str]] | None = None,
    top_fraction: float = 0.10,
    network_top_fraction: float = 0.25,
) -> pd.DataFrame:
    """Evaluate only observations already present in the locked YAML.

    ``effect_table`` contains target-minus-reference effects.  The returned
    ``evidence_class`` gives the requested exact/partial/not-supported/NE
    resolution, while ``support_status`` is mapped to the report contract.
    """

    if not 0 < top_fraction <= 0.5:
        raise ValueError("top_fraction must be in (0, 0.5]")
    if not 0 < network_top_fraction <= 0.5:
        raise ValueError("network_top_fraction must be in (0, 0.5]")
    locked = (
        load_supportive_biology(truth)
        if isinstance(truth, (str, Path))
        else dict(truth)
    )
    datasets = cast(Mapping[str, Any], locked["datasets"])
    if dataset_id not in datasets:
        raise ValueError(f"dataset {dataset_id!r} is absent from locked YAML")
    selected_mappings = (
        {"cell_types": {}, "genes": {}}
        if mappings is None
        else {
            "cell_types": dict(mappings.get("cell_types", {})),
            "genes": dict(mappings.get("genes", {})),
        }
    )
    effects = _normalize_effect_table(effect_table, mappings=selected_mappings)
    dataset_spec = cast(Mapping[str, Any], datasets[dataset_id])
    observations = cast(
        Sequence[Mapping[str, Any]], dataset_spec["expected_observations"]
    )
    rows: list[dict[str, object]] = []
    for identity_values, method in effects.groupby(
        list(IDENTITY_COLUMNS), observed=True, sort=False, dropna=False
    ):
        identity = dict(zip(IDENTITY_COLUMNS, identity_values, strict=True))
        for observation in observations:
            expected_direction = str(observation["direction"])
            expected_sign = _expected_sign(
                expected_direction, reference=reference, target=target
            )
            percentiles = _rank_percentiles(method, expected_sign)
            if "interactions" in observation:
                components = _lr_components(
                    method,
                    observation,
                    expected_sign=expected_sign,
                    percentiles=percentiles,
                    top_fraction=top_fraction,
                )
            elif "ligands" in observation:
                components = _program_components(
                    method,
                    observation,
                    expected_sign=expected_sign,
                    percentiles=percentiles,
                    top_fraction=top_fraction,
                )
            elif "pathways" in observation:
                components = _pathway_components(
                    method,
                    observation,
                    expected_sign=expected_sign,
                    percentiles=percentiles,
                    top_fraction=top_fraction,
                )
            else:
                components = _network_components(
                    method,
                    observation,
                    expected_sign=expected_sign,
                    network_top_fraction=network_top_fraction,
                )
            if not components:
                components = [
                    _component_result(
                        component_id="unsupported_observation_modality",
                        kind="unsupported",
                        covered=True,
                        estimable=False,
                        strength="not_estimable",
                        reason_code="unsupported_locked_observation_modality",
                    )
                ]
            summary = _summarize_components(
                components,
                expected_direction=expected_direction,
                reference=reference,
                target=target,
            )
            row: dict[str, object] = {
                "truth_set_id": str(locked["truth_set_id"]),
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "locked_at": str(locked["locked_at"]),
                "mapping_id": _mapping_id(selected_mappings),
                "dataset": dataset_id,
                "observation_id": str(observation["id"]),
                "expected_direction": expected_direction,
                "expected_metric": str(observation["metric"]),
                **identity,
                **summary,
                "effect_semantics": "target_minus_reference_rank_strength",
                "top_fraction_threshold": top_fraction,
                "network_top_fraction_threshold": network_top_fraction,
                "component_evidence_json": _json_components(components),
                "evidence_note": (
                    f"{summary['n_components_strong']} strong, "
                    f"{summary['n_components_directional']} directional, "
                    f"{summary['n_components_opposite']} opposite of "
                    f"{summary['n_components_expected']} locked components; "
                    "supportive silver standard, not edge-level truth"
                ),
            }
            rows.append(row)
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if set(result["evidence_class"]).difference(EVIDENCE_CLASSES):
        raise AssertionError("invalid evidence class emitted")
    if set(result["support_status"]).difference(REPORT_SUPPORT_STATUSES):
        raise AssertionError("invalid report support status emitted")
    expected_ids = {str(observation["id"]) for observation in observations}
    if set(result["observation_id"]) != expected_ids:
        raise AssertionError("evaluation did not preserve the locked observation set")
    return result


def write_biology_support(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the report-ready ``biology_support.tsv`` table."""

    missing = set(OUTPUT_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(
            f"biology support output is missing columns: {sorted(missing)}"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, OUTPUT_COLUMNS].to_csv(path, sep="\t", index=False)
    return path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table extension: {path.suffix!r}")


def read_external_long_for_effects(
    path: str | Path, *, run_id: str | None = None
) -> pd.DataFrame:
    """Read only columns required for external-long effect evaluation.

    Native universes can contain tens of millions of rows. Projecting the
    adapter contract before pandas materialization avoids retaining unused
    specificity, p-value, differential, provenance, and reason columns while
    the subject-level effect table is built.
    """

    selected = Path(path)
    if selected.suffix == ".parquet":
        filters = [("run_id", "==", run_id)] if run_id is not None else None
        return pd.read_parquet(
            selected,
            columns=list(EXTERNAL_LONG_EFFECT_COLUMNS),
            filters=filters,
        )
    if selected.suffix in {".tsv", ".txt"}:
        table = pd.read_csv(
            selected,
            sep="\t",
            usecols=list(EXTERNAL_LONG_EFFECT_COLUMNS),
        )
    elif selected.suffix == ".csv":
        table = pd.read_csv(selected, usecols=list(EXTERNAL_LONG_EFFECT_COLUMNS))
    else:
        raise ValueError(f"unsupported external-long extension: {selected.suffix!r}")
    if run_id is not None:
        table = table.loc[table["run_id"].astype(str).eq(run_id)].copy()
    return table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate locked supportive biology without real-data truth metrics"
    )
    parser.add_argument("truth_yaml", type=Path)
    parser.add_argument("input_table", type=Path)
    parser.add_argument("output_tsv", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--input-format", choices=("effect", "score", "external-long"), default="effect"
    )
    parser.add_argument("--context-key")
    parser.add_argument("--design", choices=("paired", "independent"))
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--contrast")
    parser.add_argument("--mapping-config", type=Path)
    parser.add_argument(
        "--run-id",
        help="Select one adapter run/CRYCHIC score view before effect evaluation",
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--network-top-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    args = _parser().parse_args()
    source = (
        read_external_long_for_effects(args.input_table, run_id=args.run_id)
        if args.input_format == "external-long"
        else _read_table(args.input_table)
    )
    if args.input_format == "effect":
        effects = source
    else:
        if args.design is None:
            raise ValueError("score and external-long inputs require --design")
        if args.input_format == "score":
            effects = effect_table_from_score_table(
                source,
                reference=args.reference,
                target=args.target,
                design=args.design,
                min_support=args.min_support,
                contrast=args.contrast,
            )
        else:
            if args.context_key is None:
                raise ValueError("external-long input requires --context-key")
            effects = effect_table_from_external_long(
                source,
                context_key=args.context_key,
                reference=args.reference,
                target=args.target,
                design=args.design,
                min_support=args.min_support,
                contrast=args.contrast,
            )
    evaluated = evaluate_supportive_biology(
        effects,
        args.truth_yaml,
        dataset_id=args.dataset_id,
        reference=args.reference,
        target=args.target,
        mappings=load_biology_mappings(args.mapping_config),
        top_fraction=args.top_fraction,
        network_top_fraction=args.network_top_fraction,
    )
    output = write_biology_support(evaluated, args.output_tsv)
    print(_canonical_json({"output": str(output), "rows": len(evaluated)}))


if __name__ == "__main__":
    main()
