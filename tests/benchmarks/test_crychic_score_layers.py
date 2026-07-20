from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.adapters.crychic.score_layers import (
    SCORE_LAYER_SCHEMA_VERSION,
    build_multigroup_score_layers,
    summarize_score_layer,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = {
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "fold_id": "fold-1",
        "contrast_id": "contrast-1",
        "contrast": "treated_vs_control",
        "sample_id": "sample-1",
        "subject_id": "subject-1",
        "context_id": "context-control",
        "receiver": "Receiver",
        "family_id": "family-1",
        "driver_id": "Ligand",
        "interaction_id": "lr-1",
        "mode": "state",
    }
    compact = pd.DataFrame(
        [
            {
                **base,
                "sender": sender,
                "global_sender_lr_score": 0.0,
                "status": "structural_zero",
                "reason_code": "incremental_downstream_gain_zero",
            }
            for sender in ("Sender-A", "Sender-B")
        ]
    )
    sender = pd.DataFrame(
        [
            {
                **base,
                "sender": sender_id,
                "raw_sender_evidence": evidence,
                "assignment_weight": weight,
            }
            for sender_id, evidence, weight in (
                ("Sender-A", 0.2, 0.25),
                ("Sender-B", 0.8, 0.75),
            )
        ]
    )
    lr = pd.DataFrame(
        [
            {
                **base,
                "receptor_eligible": True,
                "availability": 0.8,
                "prior_quality": 0.5,
            }
        ]
    )
    downstream = pd.DataFrame(
        [
            {
                key: base[key]
                for key in (
                    "crossfit_id",
                    "spec_id",
                    "repeat_id",
                    "fold_id",
                    "contrast_id",
                    "contrast",
                    "receiver",
                    "subject_id",
                    "family_id",
                )
            }
            | {
                "differential_effect": 0.0,
                "status": "structural_zero",
                "reason_code": "zero_receiver_contrast_structural_zero_v1",
            }
        ]
    )
    return compact, sender, lr, downstream


def test_annotate_policy_preserves_mechanism_when_strict_score_is_zero() -> None:
    result = build_multigroup_score_layers(*_frames(), policy="annotate")

    assert result["mechanistic_lr_score"].tolist() == pytest.approx([0.4, 0.4])
    assert result["mechanistic_sender_lr_score"].tolist() == pytest.approx([0.1, 0.3])
    assert result["mechanistic_sender_lr_score"].sum() == pytest.approx(0.4)
    assert result["selected_score"].tolist() == pytest.approx([0.1, 0.3])
    assert result["downstream_confirmed_sender_lr_score"].eq(0.0).all()
    assert set(result["downstream_status"]) == {"not_estimable"}
    assert set(result["downstream_reason_code"]) == {
        "receiver_contrast_below_information_floor"
    }
    assert set(result["score_layer_schema_version"]) == {SCORE_LAYER_SCHEMA_VERSION}


def test_required_policy_selects_strict_without_automatic_fallback() -> None:
    result = build_multigroup_score_layers(*_frames(), policy="required")

    assert result["selected_score"].eq(0.0).all()
    assert set(result["selected_score_status"]) == {"structural_zero"}
    assert set(result["selected_score_policy"]) == {"required"}


def test_modulation_uses_signed_support_but_missing_support_is_neutral() -> None:
    compact, sender, lr, downstream = _frames()
    downstream["differential_effect"] = -0.5
    downstream["status"] = "observed"
    downstream["reason_code"] = None
    result = build_multigroup_score_layers(
        compact,
        sender,
        lr,
        downstream,
        policy="modulate",
    )

    assert result["downstream_signed_support"].eq(-0.5).all()
    assert (result["selected_score"] < result["mechanistic_sender_lr_score"]).all()


def test_score_layer_summary_marks_all_tied_scores_as_degenerate() -> None:
    result = build_multigroup_score_layers(*_frames(), policy="required")
    summary = summarize_score_layer(
        result,
        score_column="selected_score",
        status_column="selected_score_status",
    )

    assert summary["rows"] == 2
    assert summary["nonzero_rows"] == 0
    assert summary["unique_score_count"] == 1
    assert summary["tie_fraction"] == 1.0
    assert summary["degenerate_ranking"] is True
