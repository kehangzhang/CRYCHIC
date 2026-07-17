"""Signed Track-B target-program recovery metrics for registered truth.

The evaluator operates on explicit forward and reverse activation-compatible
channels.  It does not infer a sender, receptor-specific LR edge, inhibition,
or a native NicheNet result.  AUPRC is emitted only for synthetic or
perturbation target-program truth and never for real-data biology labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

SIGNED_TRACK_B_SCHEMA: Final = "crychic-signed-track-b-evaluator-v1"
ANALYSIS_TRACK: Final = "signed_target_program"
SOURCE_AGNOSTIC_SENDER: Final = "__source_agnostic__"
FORWARD_CHANNEL: Final = "increased_activation_compatible"
REVERSE_CHANNEL: Final = "reduced_activation_compatible"
NO_DIRECTION: Final = "no_directional_program"
CHANNELS: Final = (FORWARD_CHANNEL, REVERSE_CHANNEL)
TRUTH_DIRECTIONS: Final = frozenset((*CHANNELS, NO_DIRECTION))
KNOWN_TRUTH_SCOPES: Final = frozenset(
    {"simulation_target_program_truth", "perturbation_target_program_truth"}
)
ALLOWED_STATUSES: Final = frozenset({"observed", "not_estimable", "failed", "missing"})
SCORE_SEMANTICS: Final = (
    "two_channel_activation_compatible_target_program_score_higher_is_stronger"
)
PRIMARY_METRIC: Final = "signed_target_program_macro_auprc"

TruthDirection = Literal[
    "increased_activation_compatible",
    "reduced_activation_compatible",
    "no_directional_program",
]

TRUTH_KEYS: Final = ("scenario_cell_id", "receiver", "program_id")
IDENTITY_KEYS: Final = (
    "method",
    "method_version",
    "evaluation_phase",
    "truth_set_id",
    "truth_sha256",
)
METRIC_IDENTITY_KEYS: Final = (*IDENTITY_KEYS, "truth_scope")
PREDICTION_KEYS: Final = (*IDENTITY_KEYS, "seed", *TRUTH_KEYS, "channel")


@dataclass(frozen=True)
class SignedTrackBTables:
    """Materialized audit rows and three fixed-weight metric layers."""

    materialized: pd.DataFrame
    seed_cells: pd.DataFrame
    scenario_cells: pd.DataFrame
    primary: pd.DataFrame


def _required_strings(table: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = table[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{column} must contain non-empty identifiers")
        table[column] = values.astype(str)


def _constant(table: pd.DataFrame, column: str) -> str:
    values = table[column].astype("string").dropna().astype(str).drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{column} must be constant")
    return str(values.iloc[0])


def _validate_truth(truth: pd.DataFrame) -> pd.DataFrame:
    required = {
        "truth_set_id",
        "truth_scope",
        "scenario_cell_id",
        "receiver",
        "program_id",
        "truth_direction",
    }
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"signed Track-B truth is missing: {sorted(missing)}")
    if truth.empty:
        raise ValueError("signed Track-B truth must not be empty")
    result = truth.loc[:, sorted(required)].copy()
    _required_strings(
        result,
        (
            "truth_set_id",
            "truth_scope",
            "scenario_cell_id",
            "receiver",
            "program_id",
            "truth_direction",
        ),
    )
    _constant(result, "truth_set_id")
    truth_scope = _constant(result, "truth_scope")
    if truth_scope not in KNOWN_TRUTH_SCOPES:
        raise ValueError(
            "signed Track-B AUPRC requires synthetic or perturbation "
            f"target-program truth; found {truth_scope!r}"
        )
    invalid_directions = set(result["truth_direction"]).difference(TRUTH_DIRECTIONS)
    if invalid_directions:
        raise ValueError(
            f"signed Track-B truth has invalid directions: {sorted(invalid_directions)}"
        )
    if result.duplicated(list(TRUTH_KEYS)).any():
        raise ValueError("signed Track-B truth keys must be unique")
    for receiver, receiver_truth in result.groupby(
        "receiver", observed=True, sort=False
    ):
        universes = {
            tuple(sorted(cell_truth["program_id"].astype(str).tolist()))
            for _, cell_truth in receiver_truth.groupby(
                "scenario_cell_id", observed=True, sort=False
            )
        }
        if len(universes) != 1:
            raise ValueError(
                "every scenario cell must retain the same frozen program "
                f"universe for receiver {receiver!r}"
            )
    return result.sort_values(list(TRUTH_KEYS), kind="stable").reset_index(drop=True)


def _validated_truth_sha256(truth: pd.DataFrame) -> str:
    columns = (
        "truth_set_id",
        "truth_scope",
        "scenario_cell_id",
        "receiver",
        "program_id",
        "truth_direction",
    )
    records = (
        truth.loc[:, list(columns)]
        .sort_values(list(TRUTH_KEYS), kind="stable")
        .to_dict("records")
    )
    payload = json.dumps(
        records,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def signed_track_b_truth_sha256(truth: pd.DataFrame) -> str:
    """Return the canonical digest required on signed Track-B predictions."""

    return _validated_truth_sha256(_validate_truth(truth))


def _validate_expected_seeds(expected_seeds: tuple[int, ...]) -> tuple[int, ...]:
    if not expected_seeds:
        raise ValueError("expected_seeds must not be empty")
    if any(
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
        for seed in expected_seeds
    ):
        raise ValueError("expected_seeds must contain integer identifiers")
    normalized = tuple(int(seed) for seed in expected_seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected_seeds must be unique")
    return normalized


def _missing_reason(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))
    )


def _validate_predictions(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    expected_seeds: tuple[int, ...],
) -> pd.DataFrame:
    required = {
        *PREDICTION_KEYS,
        "analysis_track",
        "sender",
        "score",
        "score_direction",
        "status",
        "reason_code",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"signed Track-B predictions are missing: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("signed Track-B predictions must not be empty")
    false_only_claims = (
        "native_nichenet_claim",
        "sender_claim",
        "lr_edge_claim",
        "supports_active_inhibition_claim",
        "formal_inference_allowed",
    )
    for column in false_only_claims:
        if column in predictions.columns and any(
            not isinstance(value, (bool, np.bool_)) or bool(value)
            for value in predictions[column]
        ):
            raise ValueError(
                f"signed Track-B predictions must set {column} to false"
            )
    result = predictions.loc[:, sorted(required)].copy()
    _required_strings(
        result,
        (
            *IDENTITY_KEYS,
            *TRUTH_KEYS,
            "analysis_track",
            "sender",
            "channel",
            "score_direction",
            "status",
        ),
    )
    if set(result["analysis_track"]) != {ANALYSIS_TRACK}:
        raise ValueError(f"analysis_track must be {ANALYSIS_TRACK}")
    if set(result["sender"]) != {SOURCE_AGNOSTIC_SENDER}:
        raise ValueError("signed Track B must remain source-agnostic")
    if set(result["score_direction"]) != {"higher"}:
        raise ValueError("signed Track-B channel scores must be higher-is-stronger")
    invalid_channels = set(result["channel"]).difference(CHANNELS)
    if invalid_channels:
        raise ValueError(f"invalid signed Track-B channels: {sorted(invalid_channels)}")
    truth_set_id = _constant(truth, "truth_set_id")
    if set(result["truth_set_id"]) != {truth_set_id}:
        raise ValueError("prediction truth_set_id does not match frozen truth")
    truth_sha256 = _validated_truth_sha256(truth)
    if set(result["truth_sha256"]) != {truth_sha256}:
        raise ValueError("prediction truth_sha256 does not match frozen truth")

    numeric_seeds = pd.to_numeric(result["seed"], errors="coerce")
    if (
        numeric_seeds.isna().any()
        or np.isinf(numeric_seeds).any()
        or not np.equal(numeric_seeds, np.floor(numeric_seeds)).all()
    ):
        raise ValueError("prediction seed must contain finite integers")
    result["seed"] = numeric_seeds.astype(np.int64)
    unexpected_seeds = set(result["seed"]).difference(expected_seeds)
    if unexpected_seeds:
        raise ValueError(
            f"predictions contain unregistered seeds: {sorted(unexpected_seeds)}"
        )
    if result.duplicated(list(PREDICTION_KEYS)).any():
        raise ValueError("signed Track-B prediction keys must be unique")

    registered = truth.loc[:, list(TRUTH_KEYS)].drop_duplicates()
    membership = result.loc[:, list(TRUTH_KEYS)].merge(
        registered.assign(_registered=True),
        on=list(TRUTH_KEYS),
        how="left",
        validate="many_to_one",
    )
    if membership["_registered"].isna().any():
        raise ValueError(
            "predictions contain programs outside the frozen truth universe"
        )

    invalid_statuses = set(result["status"]).difference(ALLOWED_STATUSES)
    if invalid_statuses:
        raise ValueError(
            "signed Track-B predictions have invalid status: "
            f"{sorted(invalid_statuses)}"
        )
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    observed = result["status"].eq("observed")
    observed_scores = result.loc[observed, "score"]
    if (
        observed_scores.isna().any()
        or np.isinf(observed_scores).any()
        or observed_scores.lt(0).any()
    ):
        raise ValueError("observed channel scores must be finite and non-negative")
    if result.loc[~observed, "score"].notna().any():
        raise ValueError("non-observed channel scores must be missing")
    observed_reasons = result.loc[observed, "reason_code"]
    if any(not _missing_reason(value) for value in observed_reasons):
        raise ValueError("observed predictions must not carry a reason_code")
    unavailable_reasons = result.loc[~observed, "reason_code"]
    if any(
        _missing_reason(value) or str(value).strip() == ""
        for value in unavailable_reasons
    ):
        raise ValueError("non-observed predictions require a reason_code")
    return result


def _expand_truth(truth: pd.DataFrame) -> pd.DataFrame:
    forward = truth.copy()
    forward["channel"] = FORWARD_CHANNEL
    reverse = truth.copy()
    reverse["channel"] = REVERSE_CHANNEL
    expanded = pd.concat([forward, reverse], ignore_index=True)
    expanded["is_positive"] = (
        expanded["truth_direction"].eq(expanded["channel"])
    ).astype(np.int8)
    return expanded


def _materialize(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    expected_seeds: tuple[int, ...],
) -> pd.DataFrame:
    identities = predictions.loc[:, list(IDENTITY_KEYS)].drop_duplicates()
    expanded_truth = _expand_truth(truth)
    expected_rows: list[pd.DataFrame] = []
    for _, identity in identities.iterrows():
        for seed in expected_seeds:
            block = expanded_truth.copy()
            for column in IDENTITY_KEYS:
                block[column] = identity[column]
            block["seed"] = seed
            expected_rows.append(block)
    expected = pd.concat(expected_rows, ignore_index=True)
    selected = predictions.loc[:, [*PREDICTION_KEYS, "score", "status", "reason_code"]]
    result = expected.merge(
        selected,
        on=list(PREDICTION_KEYS),
        how="left",
        validate="one_to_one",
    )
    absent = result["status"].isna()
    result.loc[absent, "status"] = "missing"
    result.loc[absent, "reason_code"] = "frozen_signed_program_channel_missing"
    result["analysis_track"] = ANALYSIS_TRACK
    result["sender"] = SOURCE_AGNOSTIC_SENDER
    result["score_direction"] = "higher"
    result["score_semantics"] = SCORE_SEMANTICS
    result["schema_version"] = SIGNED_TRACK_B_SCHEMA
    result["native_nichenet_claim"] = False
    result["sender_claim"] = False
    result["lr_edge_claim"] = False
    return result.sort_values(list(PREDICTION_KEYS), kind="stable").reset_index(
        drop=True
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return threshold/tie-aware non-interpolated average precision."""

    ordered = pd.DataFrame({"label": labels, "score": scores}).sort_values(
        "score", ascending=False, kind="stable"
    )
    grouped = ordered.groupby("score", sort=False, observed=True)["label"].agg(
        ["sum", "count"]
    )
    true_positive = grouped["sum"].cumsum().to_numpy(dtype=float)
    predicted_positive = grouped["count"].cumsum().to_numpy(dtype=float)
    recall = true_positive / float(labels.sum())
    precision = true_positive / predicted_positive
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))


def _unavailable_reason(group: pd.DataFrame) -> str:
    statuses = set(group["status"].astype(str))
    if "failed" in statuses:
        return "prediction_failed_in_frozen_universe"
    if "not_estimable" in statuses:
        return "prediction_not_estimable_in_frozen_universe"
    return "incomplete_frozen_program_channel_predictions"


def _seed_cell_metrics(materialized: pd.DataFrame) -> pd.DataFrame:
    group_keys = [*METRIC_IDENTITY_KEYS, "seed", "scenario_cell_id", "receiver"]
    rows: list[dict[str, object]] = []
    for key, group in materialized.groupby(group_keys, observed=True, sort=False):
        identity = dict(zip(group_keys, key, strict=True))
        n_positive = int(group["is_positive"].sum())
        n_negative = len(group) - n_positive
        observed = group["status"].eq("observed")
        if n_positive == 0:
            status = "not_estimable"
            reason = "truth_has_no_signed_positive_program"
            estimate = math.nan
        elif n_negative == 0:
            status = "not_estimable"
            reason = "truth_has_no_signed_negative_channel"
            estimate = math.nan
        elif not observed.all():
            status = "not_estimable"
            reason = _unavailable_reason(group)
            estimate = math.nan
        else:
            status = "observed"
            reason = None
            estimate = _average_precision(
                group["is_positive"].to_numpy(dtype=np.int8),
                group["score"].to_numpy(dtype=float),
            )
        rows.append(
            identity
            | {
                "metric": "signed_target_program_auprc",
                "estimate": estimate,
                "aggregation": "expanded_forward_reverse_frozen_program_channels",
                "n_registered_programs": len(group) // len(CHANNELS),
                "n_registered_channels": len(group),
                "n_truth_positive_channels": n_positive,
                "n_truth_negative_channels": n_negative,
                "n_observed_channels": int(observed.sum()),
                "status": status,
                "reason_code": reason,
                "score_semantics": SCORE_SEMANTICS,
                "schema_version": SIGNED_TRACK_B_SCHEMA,
                "native_nichenet_claim": False,
                "sender_claim": False,
                "lr_edge_claim": False,
            }
        )
    return pd.DataFrame(rows)


def _scenario_cell_metrics(
    seed_cells: pd.DataFrame,
    *,
    n_expected_seeds: int,
) -> pd.DataFrame:
    group_keys = [*METRIC_IDENTITY_KEYS, "scenario_cell_id", "receiver"]
    rows: list[dict[str, object]] = []
    for key, group in seed_cells.groupby(group_keys, observed=True, sort=False):
        identity = dict(zip(group_keys, key, strict=True))
        observed = group["status"].eq("observed")
        complete = len(group) == n_expected_seeds and observed.all()
        rows.append(
            identity
            | {
                "metric": "signed_target_program_scenario_cell_auprc",
                "estimate": (float(group["estimate"].mean()) if complete else math.nan),
                "aggregation": "equal_registered_seeds_within_scenario_receiver_cell",
                "n_registered_seeds": n_expected_seeds,
                "n_estimable_seeds": int(observed.sum()),
                "status": "observed" if complete else "not_estimable",
                "reason_code": (
                    None if complete else "not_all_registered_seeds_estimable"
                ),
                "score_semantics": SCORE_SEMANTICS,
                "schema_version": SIGNED_TRACK_B_SCHEMA,
                "native_nichenet_claim": False,
                "sender_claim": False,
                "lr_edge_claim": False,
            }
        )
    return pd.DataFrame(rows)


def _primary_metrics(
    scenario_cells: pd.DataFrame,
    *,
    n_registered_cells: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in scenario_cells.groupby(
        list(METRIC_IDENTITY_KEYS), observed=True, sort=False
    ):
        identity = dict(zip(METRIC_IDENTITY_KEYS, key, strict=True))
        observed = group["status"].eq("observed")
        complete = len(group) == n_registered_cells and observed.all()
        rows.append(
            identity
            | {
                "metric": PRIMARY_METRIC,
                "estimate": (float(group["estimate"].mean()) if complete else math.nan),
                "aggregation": (
                    "equal_registered_seeds_within_cell_then_equal_registered_"
                    "scenario_receiver_cells"
                ),
                "n_registered_scenario_receiver_cells": n_registered_cells,
                "n_estimable_scenario_receiver_cells": int(observed.sum()),
                "status": "observed" if complete else "not_estimable",
                "reason_code": (
                    None
                    if complete
                    else "not_all_registered_scenario_receiver_cells_estimable"
                ),
                "score_semantics": SCORE_SEMANTICS,
                "schema_version": SIGNED_TRACK_B_SCHEMA,
                "native_nichenet_claim": False,
                "sender_claim": False,
                "lr_edge_claim": False,
            }
        )
    return pd.DataFrame(rows)


def evaluate_signed_track_b(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    expected_seeds: tuple[int, ...],
) -> SignedTrackBTables:
    """Evaluate fixed-universe signed target-program recovery.

    Each program is expanded to forward and reverse activation-compatible
    channels.  The channel matching a non-neutral registered truth direction is
    positive; the opposite and neutral channels are negative.  Seed-level AP is
    averaged over every registered seed within a scenario/receiver cell, then
    cells receive equal weight in the primary macro-AUPRC.  No incomplete layer
    is dynamically dropped.
    """

    normalized_seeds = _validate_expected_seeds(expected_seeds)
    validated_truth = _validate_truth(truth)
    validated_predictions = _validate_predictions(
        predictions,
        validated_truth,
        expected_seeds=normalized_seeds,
    )
    materialized = _materialize(
        validated_predictions,
        validated_truth,
        expected_seeds=normalized_seeds,
    )
    seed_cells = _seed_cell_metrics(materialized)
    scenario_cells = _scenario_cell_metrics(
        seed_cells, n_expected_seeds=len(normalized_seeds)
    )
    n_registered_cells = len(
        validated_truth.loc[:, ["scenario_cell_id", "receiver"]].drop_duplicates()
    )
    primary = _primary_metrics(scenario_cells, n_registered_cells=n_registered_cells)
    return SignedTrackBTables(
        materialized=materialized,
        seed_cells=seed_cells,
        scenario_cells=scenario_cells,
        primary=primary,
    )


__all__ = [
    "ANALYSIS_TRACK",
    "CHANNELS",
    "FORWARD_CHANNEL",
    "NO_DIRECTION",
    "PRIMARY_METRIC",
    "REVERSE_CHANNEL",
    "SCORE_SEMANTICS",
    "SIGNED_TRACK_B_SCHEMA",
    "SignedTrackBTables",
    "evaluate_signed_track_b",
    "signed_track_b_truth_sha256",
]
