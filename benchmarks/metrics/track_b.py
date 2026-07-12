"""Track-B metrics for source-agnostic frozen-prior ligand activity.

Track B evaluates receiver target-program activity for candidate ligands. It is
not a native NicheNet ``predict_ligand_activities`` run, does not estimate a
sender, and does not estimate a receptor-specific or integrated LR edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmarks.adapters.common import sha256_file, write_json

TRACK_B_SUITE_SCHEMA = "crychic-track-b-suite-v1"
TRACK_B_OUTPUT_SCHEMA = "crychic-track-b-metrics-v1"
ANALYSIS_TRACK = "ligand_target_program"
SOURCE_AGNOSTIC_SENDER = "__source_agnostic__"
PROXY_SEMANTICS = "frozen_prior_activity_proxy_not_native_predict_ligand_activities"
SIGNAL_MIN_PERCENTILE = 0.80
SIGNAL_MIN_DIRECTION_CONSISTENCY = 0.625
STATUS_MAP = {
    "ok": "observed",
    "not_returned": "not_predicted",
    "resource_unavailable": "resource_unavailable",
    "unsupported_resource": "not_supported",
    "insufficient_cells": "cell_type_missing",
    "method_failed": "failed",
    "missing": "missing",
}
COMPARABLE_STATUSES = frozenset({"observed", "not_predicted"})
EDGE_KEYS = ("receiver", "interaction_id", "ligand", "target")
IDENTITY_COLUMNS = (
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


@dataclass(frozen=True)
class TrackBPrepared:
    """Ranked sample rows and subject-context values for one adapter run."""

    ranked: pd.DataFrame
    subject_context: pd.DataFrame
    identity: Mapping[str, str]
    declared_universe_size: int


@dataclass(frozen=True)
class TrackBInput:
    """One configured Track-B dataset analysis."""

    dataset: str
    path: Path
    context_key: str
    reference: str
    target: str
    design: str
    min_support: int
    scenario: str | None = None
    receiver_of_interest: str | None = None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _constant(table: pd.DataFrame, column: str) -> str:
    values = table[column].astype("string").dropna().astype(str).drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Track-B field {column!r} must be constant per run")
    return str(values.iloc[0])


def _context_values(values: pd.Series, context_key: str) -> list[str]:
    contexts: list[str] = []
    for value in values:
        parsed = json.loads(str(value))
        if not isinstance(parsed, dict) or context_key not in parsed:
            raise ValueError(
                f"context_json must encode an object containing {context_key!r}"
            )
        contexts.append(str(parsed[context_key]))
    return contexts


def prepare_track_b_long(table: pd.DataFrame, *, context_key: str) -> TrackBPrepared:
    """Validate Track B and rank candidate ligands within sample and receiver."""

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
        "sender",
        "receiver",
        "interaction_id",
        "interaction_direction",
        "ligand",
        "target",
        "score",
        "score_name",
        "score_direction",
        "status",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Track-B long table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("Track-B long table is empty")
    if _constant(table, "analysis_track") != ANALYSIS_TRACK:
        raise ValueError("Track-B metrics require analysis_track=ligand_target_program")
    if set(table["sender"].astype(str)) != {SOURCE_AGNOSTIC_SENDER}:
        raise ValueError("Track B must retain the source-agnostic sender placeholder")
    if set(table["interaction_direction"].astype(str)) != {"ligand_to_target_program"}:
        raise ValueError("Track B must use ligand_to_target_program direction")
    if table["run_id"].astype(str).nunique() != 1:
        raise ValueError("select exactly one Track-B adapter run")
    if not table["universe_member"].astype(bool).all():
        raise ValueError("Track-B metrics accept frozen-universe members only")
    for column in (
        "dataset_id",
        "method_id",
        "method_version",
        "resource_mode",
        "resource_id",
        "resource_version",
        "universe_id",
        "universe_size",
        "score_name",
        "score_direction",
    ):
        _constant(table, column)

    columns = [
        "sample_id",
        "subject_id",
        "context_json",
        "receiver",
        "interaction_id",
        "ligand",
        "target",
        "score",
        "status",
    ]
    ranked = table.loc[:, columns].copy()
    for column in (
        "sample_id",
        "subject_id",
        "receiver",
        "interaction_id",
        "ligand",
        "target",
    ):
        ranked[column] = ranked[column].astype("string").fillna("").astype(str)
    required_ids = ["sample_id", "subject_id", "receiver", "interaction_id", "ligand"]
    if ranked[required_ids].eq("").any().any():
        raise ValueError("Track-B sample and edge identifiers must be nonempty")
    ranked["context"] = _context_values(ranked["context_json"], context_key)
    duplicate_key = ["sample_id", *EDGE_KEYS]
    if ranked.duplicated(duplicate_key).any():
        raise ValueError("Track-B table has duplicate sample-receiver-ligand rows")
    declared_size = int(_constant(table, "universe_size"))
    sample_sizes = ranked.groupby("sample_id", observed=True, sort=False).size()
    if not sample_sizes.eq(declared_size).all():
        raise ValueError("every sample must materialize the Track-B frozen universe")
    universe_size = len(ranked.loc[:, EDGE_KEYS].drop_duplicates())
    if universe_size != declared_size:
        raise ValueError("Track-B declared universe_size disagrees with edge identity")

    ranked["metric_status"] = ranked["status"].astype(str).replace(STATUS_MAP)
    valid_statuses = {
        "observed",
        "not_predicted",
        "resource_unavailable",
        "not_supported",
        "cell_type_missing",
        "failed",
        "missing",
    }
    invalid = set(ranked["metric_status"]).difference(valid_statuses)
    if invalid:
        raise ValueError(f"Track-B status contains invalid values: {sorted(invalid)}")
    numeric = pd.to_numeric(ranked["score"], errors="coerce")
    observed = ranked["metric_status"].eq("observed")
    if numeric.loc[observed].isna().any() or np.isinf(numeric.fillna(0.0)).any():
        raise ValueError("observed Track-B rows require finite scores")
    if numeric.loc[~observed].notna().any():
        raise ValueError("non-observed Track-B rows require missing scores")
    direction = _constant(table, "score_direction")
    if direction not in {"higher", "lower"}:
        raise ValueError("Track-B score_direction must be higher or lower")
    oriented = numeric if direction == "higher" else -numeric
    grouping = [ranked.loc[observed, "sample_id"], ranked.loc[observed, "receiver"]]
    native_rank = (
        oriented.loc[observed]
        .groupby(grouping, observed=True)
        .rank(ascending=False, method="average")
    )
    comparable = ranked["metric_status"].isin(COMPARABLE_STATUSES)
    comparable_size = comparable.groupby(
        [ranked["sample_id"], ranked["receiver"]], observed=True
    ).transform("sum")
    ranked["comparison_rank"] = np.nan
    ranked["comparison_strength"] = np.nan
    ranked.loc[observed, "comparison_rank"] = native_rank
    ranked.loc[observed, "comparison_strength"] = (
        1.0 - (native_rank - 1.0) / comparable_size.loc[observed]
    )
    not_predicted = ranked["metric_status"].eq("not_predicted")
    ranked.loc[not_predicted, "comparison_strength"] = 0.0
    ranked.loc[not_predicted, "comparison_rank"] = comparable_size.loc[not_predicted]
    ranked["score"] = numeric

    usable = ranked.loc[ranked["comparison_strength"].notna()]
    subject_context = (
        usable.groupby(
            [*EDGE_KEYS, "subject_id", "context"],
            observed=True,
            sort=False,
            dropna=False,
        )["comparison_strength"]
        .mean()
        .reset_index()
    )
    identity = {
        "dataset": _constant(table, "dataset_id"),
        "method": _constant(table, "method_id"),
        "method_version": _constant(table, "method_version"),
        "analysis_track": ANALYSIS_TRACK,
        "resource": _constant(table, "resource_id"),
        "resource_version": _constant(table, "resource_version"),
        "resource_mode": _constant(table, "resource_mode"),
        "score_semantics": _constant(table, "score_name"),
        "universe_id": _constant(table, "universe_id"),
    }
    return TrackBPrepared(
        ranked=ranked,
        subject_context=subject_context,
        identity=identity,
        declared_universe_size=declared_size,
    )


def _edge_state(ranked: pd.DataFrame) -> pd.DataFrame:
    working = ranked.copy()
    working["_resource"] = working["metric_status"].eq("resource_unavailable")
    working["_unsupported"] = working["metric_status"].eq("not_supported")
    working["_failed"] = working["metric_status"].eq("failed")
    return (
        working.groupby(list(EDGE_KEYS), observed=True, sort=False, dropna=False)
        .agg(
            all_resource_unavailable=("_resource", "min"),
            all_not_supported=("_unsupported", "min"),
            any_failed=("_failed", "max"),
        )
        .reset_index()
    )


def _paired_effects(
    subject_context: pd.DataFrame,
    *,
    reference: str,
    target: str,
) -> pd.DataFrame:
    reference_rows = subject_context.loc[
        subject_context["context"].eq(reference),
        [*EDGE_KEYS, "subject_id", "comparison_strength"],
    ].rename(columns={"comparison_strength": "reference_strength"})
    target_rows = subject_context.loc[
        subject_context["context"].eq(target),
        [*EDGE_KEYS, "subject_id", "comparison_strength"],
    ].rename(columns={"comparison_strength": "target_strength"})
    paired = reference_rows.merge(
        target_rows,
        on=[*EDGE_KEYS, "subject_id"],
        how="inner",
        validate="one_to_one",
    )
    paired["subject_effect"] = paired["target_strength"] - paired["reference_strength"]
    rows: list[dict[str, object]] = []
    for edge, group in paired.groupby(
        list(EDGE_KEYS), observed=True, sort=False, dropna=False
    ):
        values = group["subject_effect"].to_numpy(dtype=float)
        effect = float(values.mean())
        rows.append(
            dict(zip(EDGE_KEYS, edge, strict=True))
            | {
                "effect": effect,
                "median_effect": float(np.median(values)),
                "diagnostic_standard_error": (
                    float(values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) >= 2
                    else math.nan
                ),
                "positive_direction_fraction": (
                    float(np.mean(values > 0)) if len(values) else math.nan
                ),
                "n_pairs": len(values),
                "n_reference_subjects": len(values),
                "n_target_subjects": len(values),
            }
        )
    return pd.DataFrame(rows)


def _unpaired_effects(
    subject_context: pd.DataFrame,
    *,
    reference: str,
    target: str,
) -> pd.DataFrame:
    mapping = subject_context.loc[:, ["subject_id", "context"]].drop_duplicates()
    reference_subjects = set(
        mapping.loc[mapping["context"].eq(reference), "subject_id"]
    )
    target_subjects = set(mapping.loc[mapping["context"].eq(target), "subject_id"])
    overlap = reference_subjects & target_subjects
    if overlap:
        raise ValueError(
            f"independent Track-B groups share subjects: {sorted(overlap)}"
        )
    context_summary = (
        subject_context.loc[subject_context["context"].isin((reference, target))]
        .groupby([*EDGE_KEYS, "context"], observed=True, sort=False, dropna=False)[
            "comparison_strength"
        ]
        .agg(context_mean="mean", n_subjects="size", context_sd="std")
        .reset_index()
    )
    left = context_summary.loc[
        context_summary["context"].eq(reference),
        [*EDGE_KEYS, "context_mean", "n_subjects", "context_sd"],
    ].rename(
        columns={
            "context_mean": "reference_mean",
            "n_subjects": "n_reference_subjects",
            "context_sd": "reference_sd",
        }
    )
    right = context_summary.loc[
        context_summary["context"].eq(target),
        [*EDGE_KEYS, "context_mean", "n_subjects", "context_sd"],
    ].rename(
        columns={
            "context_mean": "target_mean",
            "n_subjects": "n_target_subjects",
            "context_sd": "target_sd",
        }
    )
    result = left.merge(right, on=list(EDGE_KEYS), how="outer", validate="one_to_one")
    result["effect"] = result["target_mean"] - result["reference_mean"]
    result["median_effect"] = np.nan
    result["positive_direction_fraction"] = np.nan
    result["diagnostic_standard_error"] = np.sqrt(
        np.square(result["reference_sd"]) / result["n_reference_subjects"]
        + np.square(result["target_sd"]) / result["n_target_subjects"]
    )
    result["n_pairs"] = 0
    return result


def track_b_edge_effects(
    prepared: TrackBPrepared,
    *,
    reference: str,
    target: str,
    design: str,
    min_support: int,
) -> pd.DataFrame:
    """Compute target-reference receiver-ligand program effects."""

    if design not in {"paired", "independent"}:
        raise ValueError("Track-B design must be paired or independent")
    if isinstance(min_support, bool) or not isinstance(min_support, int):
        raise ValueError("min_support must be an integer")
    if min_support < 2:
        raise ValueError("min_support must be at least 2")
    available_contexts = set(prepared.ranked["context"])
    missing = {reference, target}.difference(available_contexts)
    if missing:
        raise ValueError(f"Track-B input is missing contexts: {sorted(missing)}")
    raw = (
        _paired_effects(prepared.subject_context, reference=reference, target=target)
        if design == "paired"
        else _unpaired_effects(
            prepared.subject_context, reference=reference, target=target
        )
    )
    universe = prepared.ranked.loc[:, EDGE_KEYS].drop_duplicates(ignore_index=True)
    effects = universe.merge(raw, on=list(EDGE_KEYS), how="left", validate="one_to_one")
    effects = effects.merge(
        _edge_state(prepared.ranked),
        on=list(EDGE_KEYS),
        how="left",
        validate="one_to_one",
    )
    if design == "paired":
        supported = effects["n_pairs"].fillna(0).ge(min_support)
        effect_semantics = "target_minus_reference_paired_receiver_ligand_rank_strength"
    else:
        supported = effects["n_reference_subjects"].fillna(0).ge(min_support) & effects[
            "n_target_subjects"
        ].fillna(0).ge(min_support)
        effect_semantics = (
            "target_minus_reference_equal_subject_receiver_ligand_rank_strength"
        )
    effects["status"] = np.select(
        [
            supported.to_numpy(dtype=bool),
            effects["all_resource_unavailable"].fillna(False).to_numpy(dtype=bool),
            effects["all_not_supported"].fillna(False).to_numpy(dtype=bool),
            effects["any_failed"].fillna(False).to_numpy(dtype=bool),
        ],
        ["exploratory", "resource_unavailable", "not_supported", "failed"],
        default="not_estimable",
    )
    effects.loc[~effects["status"].eq("exploratory"), "effect"] = np.nan
    effects["effect_rank"] = np.nan
    effects["effect_percentile"] = np.nan
    estimable = effects["status"].eq("exploratory")
    effect_rank = (
        effects.loc[estimable]
        .groupby("receiver", observed=True)["effect"]
        .rank(ascending=False, method="average")
    )
    effect_size = (
        effects.loc[estimable]
        .groupby("receiver", observed=True)["effect"]
        .transform("size")
    )
    effects.loc[estimable, "effect_rank"] = effect_rank
    effects.loc[estimable, "effect_percentile"] = np.where(
        effect_size.eq(1), 1.0, 1.0 - (effect_rank - 1.0) / (effect_size - 1.0)
    )
    positive = estimable & effects["effect"].gt(0)
    effects["positive_effect_rank"] = np.nan
    effects.loc[positive, "positive_effect_rank"] = (
        effects.loc[positive]
        .groupby("receiver", observed=True)["effect"]
        .rank(ascending=False, method="average")
    )
    effects["effect_semantics"] = effect_semantics
    effects["reference"] = reference
    effects["target_context"] = target
    effects["design"] = design
    effects["minimum_subject_support"] = min_support
    effects["proxy_semantics"] = PROXY_SEMANTICS
    for column, value in prepared.identity.items():
        effects[column] = value
    effects["reason_code"] = np.select(
        [
            effects["status"].eq("resource_unavailable").to_numpy(dtype=bool),
            effects["status"].eq("not_supported").to_numpy(dtype=bool),
            effects["status"].eq("failed").to_numpy(dtype=bool),
            effects["status"].eq("not_estimable").to_numpy(dtype=bool),
        ],
        [
            "resource_unavailable",
            "track_b_resource_not_supported",
            "track_b_run_failed",
            "insufficient_subject_or_receiver_support",
        ],
        default=None,
    )
    return effects.drop(
        columns=[
            "all_resource_unavailable",
            "all_not_supported",
            "any_failed",
        ],
        errors="ignore",
    )


def _top_keys(values: pd.Series, top_k: int) -> set[object]:
    if values.empty:
        return set()
    return set(values.nlargest(min(top_k, len(values))).index)


def _vector_stability(
    left: pd.Series,
    right: pd.Series,
    *,
    top_k: int,
) -> dict[str, object]:
    shared = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(shared) < 3:
        return {
            "n_shared_ligands": len(shared),
            "effect_spearman": math.nan,
            "direction_agreement": math.nan,
            "top_k_jaccard": math.nan,
            "status": "not_estimable",
            "reason_code": "fewer_than_three_shared_ligands",
        }
    constant = shared["left"].nunique() < 2 or shared["right"].nunique() < 2
    rho = (
        math.nan
        if constant
        else float(spearmanr(shared["left"], shared["right"]).statistic)
    )
    direction_mask = ~(shared["left"].eq(0) & shared["right"].eq(0))
    direction = (
        float(
            np.mean(
                np.sign(shared.loc[direction_mask, "left"])
                == np.sign(shared.loc[direction_mask, "right"])
            )
        )
        if direction_mask.any()
        else math.nan
    )
    left_top = _top_keys(shared["left"], top_k)
    right_top = _top_keys(shared["right"], top_k)
    union = left_top | right_top
    jaccard = len(left_top & right_top) / len(union) if union else math.nan
    estimable = np.isfinite(rho)
    return {
        "n_shared_ligands": len(shared),
        "effect_spearman": rho,
        "direction_agreement": direction,
        "top_k_jaccard": jaccard,
        "status": "observed" if estimable else "not_estimable",
        "reason_code": None if estimable else "constant_ligand_effect_vector",
    }


def _paired_stability(
    prepared: TrackBPrepared,
    *,
    reference: str,
    target: str,
    min_support: int,
    top_k: int,
) -> pd.DataFrame:
    subject_context = prepared.subject_context
    left = subject_context.loc[
        subject_context["context"].eq(reference),
        [*EDGE_KEYS, "subject_id", "comparison_strength"],
    ].rename(columns={"comparison_strength": "reference_strength"})
    right = subject_context.loc[
        subject_context["context"].eq(target),
        [*EDGE_KEYS, "subject_id", "comparison_strength"],
    ].rename(columns={"comparison_strength": "target_strength"})
    paired = left.merge(
        right,
        on=[*EDGE_KEYS, "subject_id"],
        how="inner",
        validate="one_to_one",
    )
    paired["subject_effect"] = paired["target_strength"] - paired["reference_strength"]
    rows: list[dict[str, object]] = []
    for receiver, group in paired.groupby("receiver", observed=True, sort=False):
        matrix = group.pivot_table(
            index="subject_id",
            columns="interaction_id",
            values="subject_effect",
            aggfunc="mean",
        )
        matrix = matrix.dropna(axis=1)
        enough_subjects = len(matrix) >= min_support
        for subject in matrix.index:
            if enough_subjects and len(matrix.columns) >= 3 and len(matrix) >= 3:
                held = matrix.loc[subject]
                training = matrix.drop(index=subject).mean(axis=0)
                metrics = _vector_stability(held, training, top_k=top_k)
            else:
                metrics = {
                    "n_shared_ligands": len(matrix.columns),
                    "effect_spearman": math.nan,
                    "direction_agreement": math.nan,
                    "top_k_jaccard": math.nan,
                    "status": "not_estimable",
                    "reason_code": (
                        "insufficient_paired_subjects"
                        if not enough_subjects
                        else "fewer_than_three_complete_ligands"
                    ),
                }
            rows.append(
                {
                    "receiver": str(receiver),
                    "stability_design": "paired_leave_one_subject_out",
                    "fold_id": f"held:{subject}",
                    "held_subject": str(subject),
                    "repeat_id": pd.NA,
                    "n_reference_subjects": len(matrix),
                    "n_target_subjects": len(matrix),
                    "top_k": top_k,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _derived_rng(seed: int, dataset: str, receiver: str) -> np.random.Generator:
    payload = f"{seed}:{dataset}:{receiver}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def _context_matrix(
    subject_context: pd.DataFrame,
    *,
    receiver: str,
    context: str,
) -> pd.DataFrame:
    selected = subject_context.loc[
        subject_context["receiver"].eq(receiver)
        & subject_context["context"].eq(context)
    ]
    return selected.pivot_table(
        index="subject_id",
        columns="interaction_id",
        values="comparison_strength",
        aggfunc="mean",
    )


def _unpaired_stability(
    prepared: TrackBPrepared,
    *,
    reference: str,
    target: str,
    min_support: int,
    top_k: int,
    n_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    receivers = sorted(prepared.subject_context["receiver"].astype(str).unique())
    dataset = prepared.identity["dataset"]
    for receiver in receivers:
        reference_matrix = _context_matrix(
            prepared.subject_context, receiver=receiver, context=reference
        )
        target_matrix = _context_matrix(
            prepared.subject_context, receiver=receiver, context=target
        )
        shared_columns = reference_matrix.columns.intersection(target_matrix.columns)
        reference_matrix = reference_matrix.loc[:, shared_columns].dropna(axis=1)
        target_matrix = target_matrix.loc[:, reference_matrix.columns].dropna(axis=1)
        shared_columns = reference_matrix.columns.intersection(target_matrix.columns)
        reference_matrix = reference_matrix.loc[:, shared_columns]
        target_matrix = target_matrix.loc[:, shared_columns]
        reference_half = len(reference_matrix) // 2
        target_half = len(target_matrix) // 2
        enough = (
            len(reference_matrix) >= min_support
            and len(target_matrix) >= min_support
            and reference_half >= 2
            and len(reference_matrix) - reference_half >= 2
            and target_half >= 2
            and len(target_matrix) - target_half >= 2
        )
        rng = _derived_rng(random_seed, dataset, receiver)
        for repeat in range(n_repeats):
            if enough and len(shared_columns) >= 3:
                reference_order = rng.permutation(len(reference_matrix))
                target_order = rng.permutation(len(target_matrix))
                reference_left = reference_order[:reference_half]
                reference_right = reference_order[reference_half:]
                target_left = target_order[:target_half]
                target_right = target_order[target_half:]
                left_effect = pd.Series(
                    target_matrix.iloc[target_left].mean(axis=0).to_numpy(dtype=float)
                    - reference_matrix.iloc[reference_left]
                    .mean(axis=0)
                    .to_numpy(dtype=float),
                    index=shared_columns,
                )
                right_effect = pd.Series(
                    target_matrix.iloc[target_right].mean(axis=0).to_numpy(dtype=float)
                    - reference_matrix.iloc[reference_right]
                    .mean(axis=0)
                    .to_numpy(dtype=float),
                    index=shared_columns,
                )
                metrics = _vector_stability(left_effect, right_effect, top_k=top_k)
            else:
                metrics = {
                    "n_shared_ligands": len(shared_columns),
                    "effect_spearman": math.nan,
                    "direction_agreement": math.nan,
                    "top_k_jaccard": math.nan,
                    "status": "not_estimable",
                    "reason_code": (
                        "insufficient_subjects_for_stratified_halves"
                        if not enough
                        else "fewer_than_three_complete_ligands"
                    ),
                }
            rows.append(
                {
                    "receiver": receiver,
                    "stability_design": "independent_stratified_split_half",
                    "fold_id": pd.NA,
                    "held_subject": pd.NA,
                    "repeat_id": repeat,
                    "n_reference_subjects": len(reference_matrix),
                    "n_target_subjects": len(target_matrix),
                    "top_k": top_k,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def track_b_stability(
    prepared: TrackBPrepared,
    *,
    reference: str,
    target: str,
    design: str,
    min_support: int,
    top_k: int = 5,
    n_repeats: int = 100,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired LOSO or independent split-half rank stability."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2")
    if design == "paired":
        detail = _paired_stability(
            prepared,
            reference=reference,
            target=target,
            min_support=min_support,
            top_k=top_k,
        )
    elif design == "independent":
        detail = _unpaired_stability(
            prepared,
            reference=reference,
            target=target,
            min_support=min_support,
            top_k=top_k,
            n_repeats=n_repeats,
            random_seed=random_seed,
        )
    else:
        raise ValueError("Track-B design must be paired or independent")
    if detail.empty:
        return detail, pd.DataFrame()
    for column, value in prepared.identity.items():
        detail[column] = value
    detail["proxy_semantics"] = PROXY_SEMANTICS
    summary_rows: list[dict[str, object]] = []
    for receiver, group in detail.groupby("receiver", observed=True, sort=False):
        observed = group.loc[group["status"].eq("observed")]
        estimable = not observed.empty
        summary_rows.append(
            {
                **prepared.identity,
                "receiver": str(receiver),
                "stability_design": str(group["stability_design"].iloc[0]),
                "median_effect_spearman": (
                    float(observed["effect_spearman"].median())
                    if estimable
                    else math.nan
                ),
                "median_direction_agreement": (
                    float(observed["direction_agreement"].median())
                    if estimable
                    else math.nan
                ),
                "median_top_k_jaccard": (
                    float(observed["top_k_jaccard"].median()) if estimable else math.nan
                ),
                "n_stability_folds": len(group),
                "n_stability_folds_estimable": len(observed),
                "status": "observed" if estimable else "not_estimable",
                "reason_code": (
                    None
                    if estimable
                    else str(group["reason_code"].dropna().iloc[0])
                    if group["reason_code"].notna().any()
                    else "no_estimable_stability_folds"
                ),
                "proxy_semantics": PROXY_SEMANTICS,
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _truth_bool(value: object, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"truth field {field!r} must be boolean")


def _missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def load_track_b_truth(
    path: str | Path,
    *,
    integrated_truth_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load scenario-level Track-B truth and optional integrated-edge scope."""

    truth = pd.read_csv(path, sep="\t")
    required = {
        "dataset",
        "scenario",
        "contrast",
        "expected_receiver_response",
        "truth_scope",
        "metric_scope",
        "lr_edge_truth_available",
        "reason_code",
    }
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"Track-B truth is missing columns: {sorted(missing)}")
    if truth.duplicated(["dataset", "scenario"]).any():
        raise ValueError("Track-B truth dataset/scenario rows must be unique")
    if set(truth["metric_scope"].astype(str)) != {ANALYSIS_TRACK}:
        raise ValueError("Track-B truth metric_scope must be ligand_target_program")
    truth = truth.copy()
    truth["expected_receiver_response"] = [
        _truth_bool(value, field="expected_receiver_response")
        for value in truth["expected_receiver_response"]
    ]
    truth["lr_edge_truth_available"] = [
        _truth_bool(value, field="lr_edge_truth_available")
        for value in truth["lr_edge_truth_available"]
    ]
    if truth["lr_edge_truth_available"].any():
        raise ValueError("Track B must not claim LR edge truth")
    if integrated_truth_path is None:
        truth["expected_integrated_edge"] = pd.NA
        return truth
    integrated = pd.read_csv(integrated_truth_path, sep="\t")
    needed = {"dataset_id", "scenario", "expected_integrated_edge"}
    missing_integrated = needed.difference(integrated.columns)
    if missing_integrated:
        raise ValueError(
            "integrated scenario truth is missing columns: "
            f"{sorted(missing_integrated)}"
        )
    integrated = integrated.loc[:, sorted(needed)].rename(
        columns={"dataset_id": "dataset"}
    )
    integrated["expected_integrated_edge"] = [
        _truth_bool(value, field="expected_integrated_edge")
        for value in integrated["expected_integrated_edge"]
    ]
    merged = truth.merge(
        integrated,
        on=["dataset", "scenario"],
        how="left",
        validate="one_to_one",
    )
    if merged["expected_integrated_edge"].isna().any():
        raise ValueError("integrated truth does not cover every Track-B scenario")
    return merged


def _receiver_summary(
    effects: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    receiver: str,
    scenario: str | None,
    truth_row: Mapping[str, object] | None,
) -> dict[str, object]:
    selected = effects.loc[effects["receiver"].eq(receiver)]
    estimable = selected.loc[selected["status"].eq("exploratory")]
    all_nonpositive: bool | None = (
        None if estimable.empty else bool(estimable["effect"].le(0).all())
    )
    cxcl10 = selected.loc[selected["ligand"].str.upper().eq("CXCL10")]
    cxcl10_row = cxcl10.iloc[0] if len(cxcl10) == 1 else None
    cxcl10_estimable = bool(
        cxcl10_row is not None and str(cxcl10_row["status"]) == "exploratory"
    )
    if cxcl10_estimable and cxcl10_row is not None:
        effect = float(cast(float, cxcl10_row["effect"]))
        percentile = float(cast(float, cxcl10_row["effect_percentile"]))
        direction_consistency_raw = cxcl10_row["positive_direction_fraction"]
        direction_consistency = (
            float(cast(float, direction_consistency_raw))
            if pd.notna(direction_consistency_raw)
            else math.nan
        )
        direction_gate = (
            direction_consistency >= SIGNAL_MIN_DIRECTION_CONSISTENCY
            if np.isfinite(direction_consistency)
            else True
        )
        program_positive = bool(
            effect > 0 and percentile >= SIGNAL_MIN_PERCENTILE and direction_gate
        )
        positive_rank = cxcl10_row["positive_effect_rank"]
    else:
        effect = math.nan
        percentile = math.nan
        direction_consistency = math.nan
        program_positive = False
        positive_rank = math.nan
    if estimable.empty:
        top_ligand = None
        max_effect = math.nan
    else:
        top = estimable.sort_values("effect", ascending=False, kind="stable").iloc[0]
        top_ligand = str(top["ligand"])
        max_effect = float(top["effect"])
    stability_row = stability.loc[stability["receiver"].eq(receiver)]
    if len(stability_row) == 1:
        stability_values = stability_row.iloc[0]
        stability_status = str(stability_values["status"])
    else:
        stability_values = None
        stability_status = "not_estimable"

    expected_response: bool | None = None
    expected_integrated: bool | None = None
    recovery: bool | None = None
    scope_diagnostic = "real_data_receiver_program_only"
    scope_consistent: bool | None = None
    truth_scope: str | None = None
    if truth_row is not None:
        truth_scope = "simulation"
        expected_response = bool(truth_row["expected_receiver_response"])
        raw_integrated = truth_row.get("expected_integrated_edge")
        if not _missing_scalar(raw_integrated):
            expected_integrated = bool(raw_integrated)
        recovery = (
            (program_positive if expected_response else not program_positive)
            if cxcl10_estimable
            else None
        )
        if scenario in {"target_only", "receptor_knockout"}:
            if program_positive and expected_integrated is False:
                scope_diagnostic = (
                    "program_positive_integrated_edge_negative_as_expected"
                )
                scope_consistent = True
            elif not cxcl10_estimable:
                scope_diagnostic = "scope_diagnostic_not_estimable"
                scope_consistent = None
            else:
                scope_diagnostic = "expected_program_positive_scope_not_recovered"
                scope_consistent = False
        elif scenario == "receiver_autonomous":
            scope_diagnostic = "receiver_program_without_sender_or_lr_claim"
            scope_consistent = None
        elif expected_response:
            scope_diagnostic = "receiver_program_response_expected"
        else:
            scope_diagnostic = "negative_receiver_program_expected"
    positive_rank_status = (
        "not_estimable"
        if all_nonpositive is True
        else "observed"
        if cxcl10_estimable and pd.notna(positive_rank)
        else "not_estimable"
    )
    positive_rank_reason = (
        "all_receiver_program_effects_nonpositive"
        if all_nonpositive is True
        else None
        if positive_rank_status == "observed"
        else "cxcl10_receiver_program_not_estimable_or_nonpositive"
    )
    base_identity = {
        column: str(selected[column].iloc[0]) if not selected.empty else "unknown"
        for column in IDENTITY_COLUMNS
    }
    return {
        **base_identity,
        "evaluation_scope": "simulation" if truth_row is not None else "real_data",
        "truth_scope": truth_scope,
        "scenario": scenario,
        "receiver": receiver,
        "proxy_semantics": PROXY_SEMANTICS,
        "sender_claim": False,
        "lr_or_receptor_claim": False,
        "cxcl10_effect": effect,
        "cxcl10_effect_rank": (
            float(cxcl10_row["effect_rank"])
            if cxcl10_estimable and cxcl10_row is not None
            else math.nan
        ),
        "cxcl10_effect_percentile": percentile,
        "cxcl10_positive_effect_rank": (
            float(positive_rank) if pd.notna(positive_rank) else math.nan
        ),
        "cxcl10_positive_direction_fraction": direction_consistency,
        "cxcl10_status": (
            str(cxcl10_row["status"]) if cxcl10_row is not None else "not_estimable"
        ),
        "cxcl10_reason_code": (
            str(cxcl10_row["reason_code"])
            if cxcl10_row is not None and pd.notna(cxcl10_row["reason_code"])
            else None
        ),
        "positive_rank_status": positive_rank_status,
        "positive_rank_reason_code": positive_rank_reason,
        "all_receiver_program_effects_nonpositive": all_nonpositive,
        "receiver_top_ligand": top_ligand,
        "receiver_max_effect": max_effect,
        "receiver_estimable_ligands": len(estimable),
        "receiver_program_positive": program_positive,
        "signal_min_percentile": SIGNAL_MIN_PERCENTILE,
        "signal_min_direction_consistency": SIGNAL_MIN_DIRECTION_CONSISTENCY,
        "expected_receiver_response": expected_response,
        "expected_receiver_response_recovered": recovery,
        "expected_integrated_edge": expected_integrated,
        "scope_diagnostic": scope_diagnostic,
        "scope_diagnostic_consistent": scope_consistent,
        "stability_design": (
            str(stability_values["stability_design"])
            if stability_values is not None
            else None
        ),
        "median_effect_spearman": (
            float(stability_values["median_effect_spearman"])
            if stability_values is not None
            and pd.notna(stability_values["median_effect_spearman"])
            else math.nan
        ),
        "median_direction_agreement": (
            float(stability_values["median_direction_agreement"])
            if stability_values is not None
            and pd.notna(stability_values["median_direction_agreement"])
            else math.nan
        ),
        "median_top_k_jaccard": (
            float(stability_values["median_top_k_jaccard"])
            if stability_values is not None
            and pd.notna(stability_values["median_top_k_jaccard"])
            else math.nan
        ),
        "stability_status": stability_status,
        "status": "observed" if not estimable.empty else "not_estimable",
        "reason_code": (
            None
            if not estimable.empty
            else "no_estimable_receiver_ligand_program_effects"
        ),
    }


def summarize_track_b(
    effects: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    scenario: str | None = None,
    receiver_of_interest: str | None = None,
    truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build receiver-level Track-B effect, recovery, scope, and stability rows."""

    dataset = str(effects["dataset"].iloc[0])
    truth_row: Mapping[str, object] | None = None
    if truth is not None:
        matched = truth.loc[truth["dataset"].astype(str).eq(dataset)]
        if scenario is not None:
            matched = matched.loc[matched["scenario"].astype(str).eq(scenario)]
        if len(matched) != 1:
            raise ValueError(
                f"Track-B truth must have exactly one row for {dataset}/{scenario}"
            )
        truth_row = cast(Mapping[str, object], matched.iloc[0].to_dict())
    receivers = (
        [receiver_of_interest]
        if receiver_of_interest is not None
        else sorted(effects["receiver"].astype(str).unique())
    )
    return pd.DataFrame(
        [
            _receiver_summary(
                effects,
                stability,
                receiver=receiver,
                scenario=scenario,
                truth_row=truth_row,
            )
            for receiver in receivers
        ]
    )


def simulation_truth_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert synthetic Track-B summaries to report-compatible long metrics."""

    synthetic = summary.loc[summary["evaluation_scope"].eq("simulation")]
    rows: list[dict[str, object]] = []
    metric_columns = {
        "cxcl10_receiver_program_effect": "cxcl10_effect",
        "cxcl10_receiver_program_percentile": "cxcl10_effect_percentile",
        "cxcl10_positive_effect_rank": "cxcl10_positive_effect_rank",
        "receiver_response_recovered": "expected_receiver_response_recovered",
        "scope_diagnostic_consistent": "scope_diagnostic_consistent",
        "track_b_stability_spearman": "median_effect_spearman",
    }
    for _, row in synthetic.iterrows():
        identity = {
            "dataset": row["dataset"],
            "scenario": row["scenario"],
            "method": row["method"],
            "method_version": row["method_version"],
            "analysis_track": ANALYSIS_TRACK,
            "resource": row["resource"],
            "resource_version": row["resource_version"],
            "resource_mode": row["resource_mode"],
            "receiver": row["receiver"],
            "truth_scope": "simulation",
            "proxy_semantics": PROXY_SEMANTICS,
        }
        for metric, column in metric_columns.items():
            value = row[column]
            if isinstance(value, (bool, np.bool_)):
                estimate = float(bool(value))
                status = "observed"
                reason = None
            elif pd.notna(value):
                estimate = float(value)
                status = "observed"
                reason = None
            else:
                estimate = math.nan
                status = "not_estimable"
                reason = (
                    str(row["positive_rank_reason_code"])
                    if metric == "cxcl10_positive_effect_rank"
                    else "track_b_metric_not_estimable"
                )
            rows.append(
                identity
                | {
                    "metric": metric,
                    "estimate": estimate,
                    "status": status,
                    "reason_code": reason,
                }
            )
    return pd.DataFrame(rows)


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


def _load_specification(
    path: Path, repo_root: Path
) -> tuple[dict[str, Any], list[TrackBInput]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != TRACK_B_SUITE_SCHEMA:
        raise ValueError("unsupported Track-B suite specification")
    records = raw.get("inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("Track-B suite specification has no inputs")
    inputs: list[TrackBInput] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Track-B input specifications must be mappings")
        inputs.append(
            TrackBInput(
                dataset=str(record["dataset"]),
                path=_resolve(str(record["path"]), repo_root),
                context_key=str(record["context_key"]),
                reference=str(record["reference"]),
                target=str(record["target"]),
                design=str(record["design"]),
                min_support=int(record["min_support"]),
                scenario=(
                    None if record.get("scenario") is None else str(record["scenario"])
                ),
                receiver_of_interest=(
                    None
                    if record.get("receiver_of_interest") is None
                    else str(record["receiver_of_interest"])
                ),
            )
        )
    return cast(dict[str, Any], raw), inputs


def _validate_input_artifact(path: Path, *, require_transform: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Track-B input is missing: {path}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Track-B input manifest is missing: {manifest_path}")
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise ValueError("Track-B input manifest must be an object")
    manifest = cast(dict[str, Any], manifest_raw)
    if manifest.get("status") != "complete":
        raise ValueError(f"Track-B input manifest is not complete: {manifest_path}")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("table") != path.name:
        raise ValueError("Track-B manifest output does not name the selected table")
    observed_sha = sha256_file(path)
    if output.get("sha256") != observed_sha:
        raise ValueError("Track-B table checksum disagrees with its manifest")
    method = manifest.get("method")
    if (
        not isinstance(method, dict)
        or method.get("id") != "nichenet_prior_activity"
        or method.get("native_nichenet_claim") is not False
    ):
        raise ValueError(
            "Track-B suite requires the source-agnostic NicheNet frozen-prior "
            "activity proxy, not a native predict_ligand_activities claim"
        )
    parameters = manifest.get("parameters")
    resolved = (
        parameters.get("resolved_expression_transform")
        if isinstance(parameters, dict)
        else None
    )
    if require_transform and not isinstance(resolved, str):
        raise ValueError(
            "Track-B input lacks resolved_expression_transform; wait for the "
            "normalized NicheNet rerun"
        )
    return {
        "table": str(path),
        "table_sha256": observed_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "resolved_expression_transform": resolved,
    }


def run_track_b_suite(
    specification_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run configured real and synthetic Track-B metrics and persist tables."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    spec_path = Path(specification_path).resolve()
    spec, inputs = _load_specification(spec_path, root)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Track-B output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    truth_path = _resolve(str(spec["track_b_truth"]), root)
    integrated_truth_path = _resolve(str(spec["integrated_truth"]), root)
    truth = load_track_b_truth(truth_path, integrated_truth_path=integrated_truth_path)
    top_k = int(spec.get("top_k", 5))
    n_repeats = int(spec.get("n_repeats", 100))
    random_seed = int(spec.get("random_seed", 0))
    require_transform = bool(spec.get("require_resolved_expression_transform", True))
    summaries: list[pd.DataFrame] = []
    effects_tables: list[pd.DataFrame] = []
    stability_tables: list[pd.DataFrame] = []
    input_manifest: list[dict[str, Any]] = []
    for item in inputs:
        artifact = _validate_input_artifact(
            item.path, require_transform=require_transform
        )
        table = pd.read_parquet(item.path)
        prepared = prepare_track_b_long(table, context_key=item.context_key)
        if prepared.identity["dataset"] != item.dataset:
            raise ValueError(
                f"Track-B configured dataset disagrees with adapter: {item.dataset}"
            )
        effects = track_b_edge_effects(
            prepared,
            reference=item.reference,
            target=item.target,
            design=item.design,
            min_support=item.min_support,
        )
        stability_detail, stability_summary = track_b_stability(
            prepared,
            reference=item.reference,
            target=item.target,
            design=item.design,
            min_support=item.min_support,
            top_k=top_k,
            n_repeats=n_repeats,
            random_seed=random_seed,
        )
        summary = summarize_track_b(
            effects,
            stability_summary,
            scenario=item.scenario,
            receiver_of_interest=item.receiver_of_interest,
            truth=truth if item.scenario is not None else None,
        )
        effects["scenario"] = item.scenario
        stability_detail["scenario"] = item.scenario
        summaries.append(summary)
        effects_tables.append(effects)
        stability_tables.append(stability_detail)
        input_manifest.append(
            artifact
            | {
                "dataset": item.dataset,
                "scenario": item.scenario,
                "design": item.design,
                "reference": item.reference,
                "target": item.target,
            }
        )
    summary_table = pd.concat(summaries, ignore_index=True)
    effects_table = pd.concat(effects_tables, ignore_index=True)
    stability_table = pd.concat(stability_tables, ignore_index=True)
    simulation_table = simulation_truth_table(summary_table)
    effect_detail = effects_table.copy()
    effect_detail["record_type"] = "receiver_ligand_effect"
    stability_detail = stability_table.copy()
    stability_detail["record_type"] = "stability_fold"
    detail_table = pd.concat(
        [effect_detail, stability_detail], ignore_index=True, sort=False
    )
    paths = {
        "summary": output / "track_b_summary.tsv",
        "detail": output / "track_b_detail.tsv",
        "effects": output / "track_b_effects.tsv",
        "stability": output / "track_b_stability.tsv",
        "simulation_truth": output / "track_b_simulation_truth.tsv",
    }
    summary_table.to_csv(paths["summary"], sep="\t", index=False)
    detail_table.to_csv(paths["detail"], sep="\t", index=False)
    effects_table.to_csv(paths["effects"], sep="\t", index=False)
    stability_table.to_csv(paths["stability"], sep="\t", index=False)
    simulation_table.to_csv(paths["simulation_truth"], sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": TRACK_B_OUTPUT_SCHEMA,
        "specification": str(spec_path),
        "specification_sha256": sha256_file(spec_path),
        "proxy_semantics": PROXY_SEMANTICS,
        "sender_claim": False,
        "lr_or_receptor_claim": False,
        "settings": {
            "top_k": top_k,
            "n_repeats": n_repeats,
            "random_seed": random_seed,
            "signal_min_percentile": SIGNAL_MIN_PERCENTILE,
            "signal_min_direction_consistency": (SIGNAL_MIN_DIRECTION_CONSISTENCY),
        },
        "inputs": input_manifest,
        "outputs": {
            name: {
                "path": path.name,
                "rows": len(
                    {
                        "summary": summary_table,
                        "detail": detail_table,
                        "effects": effects_table,
                        "stability": stability_table,
                        "simulation_truth": simulation_table,
                    }[name]
                ),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-agnostic Track-B receiver-program effects and stability"
        )
    )
    parser.add_argument("specification", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_track_b_suite(
        args.specification,
        args.output_dir,
        overwrite=args.overwrite,
        repo_root=args.repo_root,
    )
    print(
        _canonical_json(
            {
                "status": "complete",
                "output": str(Path(args.output_dir).resolve()),
                "rows": manifest["outputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
