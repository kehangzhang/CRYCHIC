"""Adapt native common-scale CRYCHIC scores to signed Track-B rows."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from crychic.workflow import DirectionalTargetProgramScoreCollection

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METHOD = "crychic"
_OUTPUT_COLUMNS = (
    "method",
    "method_version",
    "evaluation_phase",
    "truth_set_id",
    "truth_sha256",
    "seed",
    "scenario_cell_id",
    "collection_id",
    "crossfit_id",
    "crossfit_spec_id",
    "receiver_universe_id",
    "receiver_axis_id",
    "receiver_training_support_ids",
    "pair_spec_id",
    "target_program_universe_id",
    "score_spec_id",
    "effect_result_id",
    "receiver",
    "program_id",
    "channel",
    "analysis_track",
    "sender",
    "score",
    "score_direction",
    "status",
    "reason_code",
    "effect_backend",
    "score_method",
    "value_scale",
    "supports_active_inhibition_claim",
    "formal_inference_allowed",
    "native_nichenet_claim",
    "sender_claim",
    "lr_edge_claim",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def adapt_directional_target_program_scores_to_signed_track_b(
    collection: DirectionalTargetProgramScoreCollection,
    *,
    method_version: str,
    evaluation_phase: str,
    truth_set_id: str,
    truth_sha256: str,
    seed: int,
    scenario_cell_id: str,
) -> pd.DataFrame:
    """Attach campaign identity without changing the frozen score universe."""

    if not isinstance(collection, DirectionalTargetProgramScoreCollection):
        raise TypeError("collection must be DirectionalTargetProgramScoreCollection")
    method_version = _name(method_version, field_name="method_version")
    evaluation_phase = _name(evaluation_phase, field_name="evaluation_phase")
    truth_set_id = _name(truth_set_id, field_name="truth_set_id")
    truth_sha256 = _name(truth_sha256, field_name="truth_sha256")
    scenario_cell_id = _name(scenario_cell_id, field_name="scenario_cell_id")
    if _SHA256.fullmatch(truth_sha256) is None:
        raise ValueError("truth_sha256 must be a lowercase SHA-256 digest")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    source = collection.scores
    result = source.loc[
        :,
        [
            "crossfit_id",
            "receiver_universe_id",
            "receiver_axis_id",
            "receiver_training_support_ids",
            "pair_spec_id",
            "target_program_universe_id",
            "score_spec_id",
            "effect_result_id",
            "receiver",
            "program_id",
            "channel",
            "analysis_track",
            "sender",
            "score",
            "score_direction",
            "status",
            "reason_code",
            "effect_backend",
            "score_method",
            "value_scale",
            "supports_active_inhibition_claim",
            "formal_inference_allowed",
        ],
    ].copy()
    result["method"] = _METHOD
    result["method_version"] = method_version
    result["evaluation_phase"] = evaluation_phase
    result["truth_set_id"] = truth_set_id
    result["truth_sha256"] = truth_sha256
    result["seed"] = int(seed)
    result["scenario_cell_id"] = scenario_cell_id
    result["collection_id"] = collection.collection_id
    result["crossfit_spec_id"] = collection.crossfit_spec_id
    result["native_nichenet_claim"] = False
    result["sender_claim"] = False
    result["lr_edge_claim"] = False
    result = result.loc[:, list(_OUTPUT_COLUMNS)]
    if tuple(result.columns) != _OUTPUT_COLUMNS:
        raise RuntimeError("signed Track-B adapter emitted an invalid column order")
    sorted_result: pd.DataFrame = result.sort_values(
        ["seed", "scenario_cell_id", "receiver", "program_id", "channel"],
        kind="stable",
        ignore_index=True,
    )
    return sorted_result


__all__ = ["adapt_directional_target_program_scores_to_signed_track_b"]
