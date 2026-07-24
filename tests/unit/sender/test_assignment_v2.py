from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tests.support.sample_edge_v2 import sample_edge_scores

from crychic.scoring import SampleEdgeScoreV2
from crychic.sender import (
    EBShrunkenCouplingV2Spec,
    SenderAttributionV2Spec,
    SenderV2CalibrationStatus,
    apply_sender_attribution_v2,
    fit_eb_shrunken_coupling_v2,
    fit_sender_attribution_v2,
    sender_attribution_v2_application_id,
)


def _training_scores(*, condition_label: str = "training") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    residuals = (-0.3, -0.2, -0.1, 0.0, 0.0, 0.1, 0.2, 0.3)
    for index, residual in enumerate(residuals):
        for sender, baseline in (("sender-a", 2.0), ("sender-b", 1.0)):
            rows.append(
                {
                    "sample_id": f"train-sample-{index}",
                    "subject_id": f"train-subject-{index}",
                    "condition": condition_label,
                    "sender": sender,
                    "receiver": "receiver",
                    "interaction_id": "interaction-1",
                    "sender_detection_raw": baseline + residual,
                    "parent_mean_raw": 1.5 + residual,
                    "parent_peak_raw": 2.0 + residual,
                    "parent_total_raw": 3.0 + 2.0 * residual,
                    "status": "observed",
                }
            )
    return pd.DataFrame(rows)


def _functional(*, minimum_subjects: int = 4):
    return fit_sender_attribution_v2(
        _training_scores(),
        fold_id="fold-1",
        training_subject_ids=tuple(f"train-subject-{index}" for index in range(8)),
        training_input_digest="training-input-digest",
        activity_transform_id="transform-id",
        spec=SenderAttributionV2Spec(minimum_calibration_subjects=minimum_subjects),
    )


def _coupling_functional():
    receiver = np.asarray([-1.0, -0.7, -0.3, -0.1, 0.2, 0.4, 0.8, 1.1])
    decoy = np.asarray([0.8, -0.9, 0.5, -0.4, 0.2, -0.1, -0.6, 0.7])
    rows = [
        {
            "subject_id": f"train-subject-{index}",
            "sender": sender,
            "receiver": "receiver",
            "interaction_id": "interaction-1",
            "sender_effect": float(values[index]),
            "receiver_effect": float(receiver[index]),
        }
        for sender, values in (("sender-a", receiver), ("sender-b", decoy))
        for index in range(8)
    ]
    return fit_eb_shrunken_coupling_v2(
        pd.DataFrame(rows),
        fold_id="fold-1",
        training_subject_ids=tuple(f"train-subject-{index}" for index in range(8)),
        training_input_digest="training-input-digest",
        sender_activity_transform_id="transform-id",
        receiver_program_functional_id="program-functional-id",
        spec=EBShrunkenCouplingV2Spec(
            minimum_subjects=4,
            minimum_observed_edges_for_eb=2,
            attribution_coupling_weight=0.5,
        ),
    )


def _unattributed_scores() -> SampleEdgeScoreV2:
    original = sample_edge_scores()
    table = original.table.copy(deep=True)
    table["active_probability"] = np.nan
    table["occurrence_status"] = "not_computed"
    table["occurrence_reason_code"] = "active_probability_not_computed"
    table["occurrence_functional_id"] = None
    table["sender_attribution"] = np.nan
    table["null_sender_attribution"] = np.nan
    table["attribution_entropy"] = np.nan
    table["attribution_status"] = "not_computed"
    table["attribution_reason_code"] = "sender_attribution_not_computed"
    table["attribution_functional_id"] = None
    return SampleEdgeScoreV2(table=table, provenance=original.provenance)


def test_fit_is_outcome_agnostic_and_row_order_deterministic() -> None:
    spec = SenderAttributionV2Spec(minimum_calibration_subjects=4)
    kwargs = {
        "fold_id": "fold-1",
        "training_subject_ids": tuple(f"train-subject-{index}" for index in range(8)),
        "training_input_digest": "training-input-digest",
        "activity_transform_id": "transform-id",
        "spec": spec,
    }

    first = fit_sender_attribution_v2(_training_scores(condition_label="a"), **kwargs)
    second = fit_sender_attribution_v2(
        _training_scores(condition_label="b").sample(frac=1.0, random_state=7),
        **kwargs,
    )

    assert first.functional_id == second.functional_id
    assert first.outcome_agnostic
    assert not first.formal_inference_allowed


def test_null_sender_attribution_conserves_only_conditional_mass() -> None:
    scores = _unattributed_scores()
    functional = _functional()

    table = apply_sender_attribution_v2(
        functional,
        scores.table,
        activity_transform_id="transform-id",
    )
    result = SampleEdgeScoreV2(table=table, provenance=scores.provenance).table

    assert result["sender_detection_raw"].tolist() == [2.0, 1.0]
    assert (
        result.loc[result["sender"].eq("sender-a"), "sender_attribution"].iloc[0]
        > result.loc[result["sender"].eq("sender-b"), "sender_attribution"].iloc[0]
    )
    sender_mass = float(result["sender_attribution"].sum())
    null_mass = float(result["null_sender_attribution"].iloc[0])
    assert sender_mass + null_mass == pytest.approx(1.0)
    assert sender_mass == pytest.approx(result["active_probability"].iloc[0])
    assert 0.0 <= result["attribution_entropy"].iloc[0] <= 1.0
    assert set(result["attribution_functional_id"]) == {functional.functional_id}
    assert set(result["occurrence_functional_id"]) == {functional.functional_id}


def test_parent_activity_probability_is_monotone_in_heldout_score() -> None:
    functional = _functional()
    scores = _unattributed_scores()
    low = scores.table.copy(deep=True)
    high = scores.table.copy(deep=True)
    for column in ("parent_peak_raw", "parent_total_raw", "parent_mean_raw"):
        low[column] = 0.2
        high[column] = 4.0

    low_result = apply_sender_attribution_v2(
        functional, low, activity_transform_id="transform-id"
    )
    high_result = apply_sender_attribution_v2(
        functional, high, activity_transform_id="transform-id"
    )

    assert (
        low_result["active_probability"].iloc[0]
        < high_result["active_probability"].iloc[0]
    )
    assert (
        low_result["null_sender_attribution"].iloc[0]
        > high_result["null_sender_attribution"].iloc[0]
    )


def test_eb_coupling_changes_attribution_not_raw_detection() -> None:
    functional = _functional()
    coupling = _coupling_functional()
    scores = _unattributed_scores()
    baseline = apply_sender_attribution_v2(
        functional,
        scores.table,
        activity_transform_id="transform-id",
    ).set_index("sender")
    coupled = apply_sender_attribution_v2(
        functional,
        scores.table,
        activity_transform_id="transform-id",
        coupling_functional=coupling,
    )
    result = SampleEdgeScoreV2(
        table=coupled, provenance=scores.provenance
    ).table.set_index("sender")

    assert result["sender_detection_raw"].equals(
        scores.table.set_index("sender")["sender_detection_raw"]
    )
    assert (
        result.loc["sender-a", "coupling_prior"]
        > result.loc["sender-b", "coupling_prior"]
    )
    assert (
        result.loc["sender-a", "sender_attribution"]
        > baseline.loc["sender-a", "sender_attribution"]
    )
    assert set(result["coupling_functional_id"]) == {coupling.functional_id}
    assert set(result["attribution_functional_id"]) == {
        sender_attribution_v2_application_id(functional, coupling)
    }


def test_missing_coupling_record_is_annotation_not_hard_filter() -> None:
    functional = _functional()
    coupling = _coupling_functional()
    partial_coupling = replace(
        coupling,
        records=tuple(
            record for record in coupling.records if record.sender == "sender-a"
        ),
    )
    scores = _unattributed_scores()

    output = apply_sender_attribution_v2(
        functional,
        scores.table,
        activity_transform_id="transform-id",
        coupling_functional=partial_coupling,
    )
    result = SampleEdgeScoreV2(
        table=output, provenance=scores.provenance
    ).table.set_index("sender")

    assert result.loc["sender-b", "coupling_status"] == "not_estimable"
    assert pd.isna(result.loc["sender-b", "coupling_prior"])
    assert result.loc["sender-b", "attribution_status"] == "observed"
    assert pd.notna(result.loc["sender-b", "sender_attribution"])


def test_missing_sender_does_not_invalidate_observed_candidate() -> None:
    functional = _functional()
    scores = _unattributed_scores()
    table = scores.table.copy(deep=True)
    missing = table["sender"].eq("sender-b")
    table.loc[missing, "sender_detection_raw"] = np.nan
    table.loc[missing, "ligand_activity_raw"] = np.nan
    table.loc[missing, "status"] = "not_estimable"
    table.loc[missing, "reason_code"] = "sender_not_measured"
    table.loc[missing, "coverage_status"] = "not_measured_or_not_estimable"
    table["effective_candidate_count"] = 1
    for column in ("parent_peak_raw", "parent_total_raw", "parent_mean_raw"):
        table[column] = 2.0
    scores = SampleEdgeScoreV2(table=table, provenance=scores.provenance)

    output = apply_sender_attribution_v2(
        functional, scores.table, activity_transform_id="transform-id"
    )
    result = SampleEdgeScoreV2(table=output, provenance=scores.provenance).table

    observed = result.loc[~missing].iloc[0]
    unavailable = result.loc[missing].iloc[0]
    assert observed["attribution_status"] == "partial"
    assert pd.notna(observed["sender_attribution"])
    assert unavailable["attribution_status"] == "not_estimable"
    assert pd.isna(unavailable["sender_attribution"])
    assert observed["sender_attribution"] + observed["null_sender_attribution"] == (
        pytest.approx(1.0)
    )


def test_insufficient_training_support_is_typed_not_estimable() -> None:
    training = _training_scores().loc[
        lambda table: table["subject_id"].isin(
            {"train-subject-0", "train-subject-1", "train-subject-2", "train-subject-3"}
        )
    ]
    functional = fit_sender_attribution_v2(
        training,
        fold_id="fold-1",
        training_subject_ids=tuple(sorted(training["subject_id"].unique())),
        training_input_digest="training-input-digest",
        activity_transform_id="transform-id",
        spec=SenderAttributionV2Spec(minimum_calibration_subjects=6),
    )

    assert all(
        item.status is SenderV2CalibrationStatus.NOT_ESTIMABLE
        for item in functional.parent_calibrations
    )
    output = apply_sender_attribution_v2(
        functional,
        _unattributed_scores().table,
        activity_transform_id="transform-id",
    )
    assert set(output["attribution_status"]) == {"not_estimable"}
    assert output["sender_attribution"].isna().all()
    assert output["null_sender_attribution"].isna().all()


def test_application_refuses_fold_transform_and_head_overwrite_mismatch() -> None:
    functional = _functional()
    scores = _unattributed_scores()

    with pytest.raises(ValueError, match="transform does not match"):
        apply_sender_attribution_v2(
            functional,
            scores.table,
            activity_transform_id="other-transform",
        )

    wrong_fold = scores.table.assign(fold_id="fold-other")
    with pytest.raises(ValueError, match="fold_id does not match"):
        apply_sender_attribution_v2(
            functional,
            wrong_fold,
            activity_transform_id="transform-id",
        )

    already_applied = scores.table.copy(deep=True)
    already_applied["occurrence_status"] = "not_estimable"
    with pytest.raises(ValueError, match="refuses to overwrite occurrence_status"):
        apply_sender_attribution_v2(
            functional,
            already_applied,
            activity_transform_id="transform-id",
        )
