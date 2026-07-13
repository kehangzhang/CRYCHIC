from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics.signed_track_b import (
    FORWARD_CHANNEL,
    PRIMARY_METRIC,
    REVERSE_CHANNEL,
    evaluate_signed_track_b,
    signed_track_b_truth_sha256,
)


def _truth() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cells = {
        "cell_a": [
            ("up", FORWARD_CHANNEL),
            ("down", REVERSE_CHANNEL),
            ("neutral", "no_directional_program"),
        ],
        "cell_b": [
            ("up", FORWARD_CHANNEL),
            ("down", "no_directional_program"),
            ("neutral", "no_directional_program"),
        ],
    }
    for cell, programs in cells.items():
        for program, direction in programs:
            rows.append(
                {
                    "truth_set_id": "signed-truth-v1",
                    "truth_scope": "simulation_target_program_truth",
                    "scenario_cell_id": cell,
                    "receiver": "Receiver",
                    "program_id": program,
                    "truth_direction": direction,
                }
            )
    return pd.DataFrame(rows)


def _predictions(
    truth: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = (11, 12),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    truth_sha256 = signed_track_b_truth_sha256(truth)
    for seed in seeds:
        for record in truth.to_dict("records"):
            direction = str(record["truth_direction"])
            for channel in (FORWARD_CHANNEL, REVERSE_CHANNEL):
                if direction == channel:
                    score = 0.9
                elif direction == "no_directional_program":
                    score = 0.2
                else:
                    score = 0.1
                rows.append(
                    {
                        "method": "candidate",
                        "method_version": "1",
                        "evaluation_phase": "development",
                        "truth_set_id": "signed-truth-v1",
                        "truth_sha256": truth_sha256,
                        "seed": seed,
                        "scenario_cell_id": record["scenario_cell_id"],
                        "receiver": record["receiver"],
                        "program_id": record["program_id"],
                        "channel": channel,
                        "analysis_track": "signed_target_program",
                        "sender": "__source_agnostic__",
                        "score": score,
                        "score_direction": "higher",
                        "status": "observed",
                        "reason_code": None,
                    }
                )
    return pd.DataFrame(rows)


def test_perfect_forward_reverse_recovery_has_macro_auprc_one() -> None:
    truth = _truth()
    tables = evaluate_signed_track_b(
        _predictions(truth), truth, expected_seeds=(11, 12)
    )

    assert set(tables.seed_cells["estimate"]) == {1.0}
    assert set(tables.scenario_cells["estimate"]) == {1.0}
    primary = tables.primary.iloc[0]
    assert primary["metric"] == PRIMARY_METRIC
    assert primary["estimate"] == pytest.approx(1.0)
    assert primary["aggregation"] == (
        "equal_registered_seeds_within_cell_then_equal_registered_"
        "scenario_receiver_cells"
    )
    assert primary["status"] == "observed"
    assert primary["truth_scope"] == "simulation_target_program_truth"
    assert primary["truth_sha256"] == signed_track_b_truth_sha256(truth)
    assert primary["schema_version"] == "crychic-signed-track-b-evaluator-v1"
    assert not bool(primary["native_nichenet_claim"])
    assert not bool(primary["sender_claim"])
    assert not bool(primary["lr_edge_claim"])
    assert not any(
        token in column.casefold()
        for column in tables.primary.columns
        for token in ("p_value", "q_value", "fdr")
    )


def test_wrong_direction_channels_are_false_positives_not_relabelled() -> None:
    truth = _truth().loc[lambda table: table["scenario_cell_id"].eq("cell_a")]
    predictions = _predictions(truth, seeds=(11,))
    signed = truth.set_index("program_id")["truth_direction"].to_dict()
    for index, row in predictions.iterrows():
        direction = signed[str(row["program_id"])]
        if direction != "no_directional_program":
            predictions.loc[index, "score"] = (
                0.1 if row["channel"] == direction else 0.9
            )

    tables = evaluate_signed_track_b(predictions, truth, expected_seeds=(11,))

    # Both registered positives are tied below their two opposite-direction
    # false positives and the neutral channels, so AP equals prevalence.
    assert tables.seed_cells.loc[0, "estimate"] == pytest.approx(2 / 6)


def test_tied_channel_scores_use_tie_inclusive_thresholds() -> None:
    truth = _truth().loc[lambda table: table["scenario_cell_id"].eq("cell_a")]
    predictions = _predictions(truth, seeds=(11,))
    predictions["score"] = 1.0

    tables = evaluate_signed_track_b(predictions, truth, expected_seeds=(11,))

    assert tables.seed_cells.loc[0, "estimate"] == pytest.approx(2 / 6)


def test_missing_frozen_channel_propagates_ne_without_zero_imputation() -> None:
    truth = _truth()
    predictions = _predictions(truth)
    missing = (
        predictions["seed"].eq(12)
        & predictions["scenario_cell_id"].eq("cell_b")
        & predictions["program_id"].eq("up")
        & predictions["channel"].eq(REVERSE_CHANNEL)
    )
    predictions = predictions.loc[~missing]

    tables = evaluate_signed_track_b(predictions, truth, expected_seeds=(11, 12))

    materialized = tables.materialized.loc[
        tables.materialized["seed"].eq(12)
        & tables.materialized["scenario_cell_id"].eq("cell_b")
        & tables.materialized["program_id"].eq("up")
        & tables.materialized["channel"].eq(REVERSE_CHANNEL)
    ].iloc[0]
    assert materialized["status"] == "missing"
    assert materialized["reason_code"] == "frozen_signed_program_channel_missing"
    assert math.isnan(float(materialized["score"]))
    cell_b = tables.scenario_cells.loc[
        tables.scenario_cells["scenario_cell_id"].eq("cell_b")
    ].iloc[0]
    assert cell_b["status"] == "not_estimable"
    assert cell_b["n_estimable_seeds"] == 1
    assert tables.primary.loc[0, "status"] == "not_estimable"
    assert math.isnan(float(tables.primary.loc[0, "estimate"]))


def test_primary_weights_registered_cells_not_positive_channel_count() -> None:
    truth = _truth()
    predictions = _predictions(truth, seeds=(11,))
    cell_b = predictions["scenario_cell_id"].eq("cell_b")
    predictions.loc[cell_b, "score"] = 1.0

    tables = evaluate_signed_track_b(predictions, truth, expected_seeds=(11,))
    by_cell = tables.scenario_cells.set_index("scenario_cell_id")["estimate"]

    assert by_cell["cell_a"] == pytest.approx(1.0)
    # One positive among six tied expanded channels. The primary still gives
    # each cell one half of the weight despite unequal positive counts.
    assert by_cell["cell_b"] == pytest.approx(1 / 6)
    assert tables.primary.loc[0, "estimate"] == pytest.approx((1.0 + 1 / 6) / 2)


def test_all_neutral_truth_cell_is_explicitly_not_estimable() -> None:
    truth = _truth().loc[lambda table: table["scenario_cell_id"].eq("cell_b")].copy()
    truth["truth_direction"] = "no_directional_program"
    predictions = _predictions(truth, seeds=(11,))

    tables = evaluate_signed_track_b(predictions, truth, expected_seeds=(11,))

    assert tables.seed_cells.loc[0, "status"] == "not_estimable"
    assert tables.seed_cells.loc[0, "reason_code"] == (
        "truth_has_no_signed_positive_program"
    )
    assert tables.primary.loc[0, "status"] == "not_estimable"


def test_scenario_specific_program_universe_is_rejected() -> None:
    truth = _truth()
    changed = truth["scenario_cell_id"].eq("cell_b") & truth["program_id"].eq("neutral")
    truth.loc[changed, "program_id"] = "posthoc_program"

    with pytest.raises(ValueError, match="same frozen program universe"):
        evaluate_signed_track_b(_predictions(_truth()), truth, expected_seeds=(11, 12))


def test_truth_content_change_under_same_id_is_rejected_by_digest() -> None:
    truth = _truth()
    predictions = _predictions(truth)
    changed = truth["scenario_cell_id"].eq("cell_b") & truth["program_id"].eq("down")
    truth.loc[changed, "truth_direction"] = REVERSE_CHANNEL

    with pytest.raises(ValueError, match="truth_sha256"):
        evaluate_signed_track_b(predictions, truth, expected_seeds=(11, 12))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.assign(seed=99), "unregistered seeds"),
        (lambda table: table.assign(sender="Sender"), "source-agnostic"),
        (
            lambda table: table.assign(channel="signed_coefficient"),
            "invalid signed Track-B channels",
        ),
        (
            lambda table: table.assign(analysis_track="lr_stlr"),
            "analysis_track",
        ),
    ],
)
def test_prediction_contract_rejects_scope_drift(
    mutation: object, message: str
) -> None:
    truth = _truth()
    predictions = mutation(_predictions(truth))  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        evaluate_signed_track_b(predictions, truth, expected_seeds=(11, 12))


def test_nonobserved_scores_are_missing_and_carry_reasons() -> None:
    truth = _truth()
    predictions = _predictions(truth)
    predictions.loc[0, "status"] = "not_estimable"
    predictions.loc[0, "reason_code"] = "channel_fit_not_estimable"
    predictions.loc[0, "score"] = np.nan

    tables = evaluate_signed_track_b(predictions, truth, expected_seeds=(11, 12))
    assert (
        tables.seed_cells.loc[
            tables.seed_cells["seed"].eq(11)
            & tables.seed_cells["scenario_cell_id"].eq("cell_a"),
            "reason_code",
        ].iloc[0]
        == "prediction_not_estimable_in_frozen_universe"
    )

    predictions.loc[0, "score"] = 0.0
    with pytest.raises(ValueError, match="non-observed channel scores"):
        evaluate_signed_track_b(predictions, truth, expected_seeds=(11, 12))


def test_real_data_labels_cannot_authorize_auprc() -> None:
    truth = _truth()
    truth["truth_scope"] = "supportive_real_data_biology"

    with pytest.raises(ValueError, match="synthetic or perturbation"):
        evaluate_signed_track_b(_predictions(_truth()), truth, expected_seeds=(11, 12))
