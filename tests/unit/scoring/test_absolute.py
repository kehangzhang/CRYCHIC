from __future__ import annotations

import numpy as np
import pandas as pd

from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
    InteractionFilterPolicy,
)
from crychic.scoring import (
    ABSOLUTE_ACTIVITY_HEAD_COLUMNS,
    ABSOLUTE_ACTIVITY_SCORE_VERSION,
    build_absolute_activity_heads,
)
from crychic.sender import SenderEvidenceParameters, assign_senders


def _availability(
    *, include_decoy: bool = False, low_evidence: bool = False
) -> BatchAvailability:
    sender_values = {
        "strong": 0.05 if low_evidence else 0.8,
        "weak": 0.04 if low_evidence else 0.1,
    }
    if include_decoy:
        sender_values["decoy"] = 0.03 if low_evidence else 0.05
    receptor = 0.05 if low_evidence else 0.6
    rows: list[dict[str, object]] = []
    for index in range(3):
        for sender, ligand in sender_values.items():
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "subject_id": f"subject-{index}",
                    "context_id": "context-a",
                    "condition": "a",
                    "sender": sender,
                    "receiver": "receiver",
                    "interaction_id": "lr-1",
                    "source_interaction_id": "L_R",
                    "ligand": "L",
                    "receptor": "R",
                    "pathway": "test",
                    "ligand_availability": ligand,
                    "receptor_availability": receptor,
                    "ligand_absolute_evidence": ligand,
                    "receptor_absolute_evidence": receptor,
                    "absolute_lr_activity": 0.5 * (ligand + receptor),
                    "sender_proportion": 1.0 / len(sender_values),
                    "receiver_proportion": 1.0,
                    "availability_state": ligand * receptor,
                    "availability_ecosystem": ligand * receptor,
                    "state_status": "observed",
                    "state_reason_code": None,
                    "ecosystem_status": "observed",
                    "ecosystem_reason_code": None,
                    "ecosystem_label": "capture_weighted_ecosystem_proxy",
                }
            )
    subjects = tuple(f"subject-{index}" for index in range(3))
    universe = FrozenInteractionUniverse(
        interaction_ids=("lr-1",),
        training_subject_ids=subjects,
        resource_id="fixture",
        resource_version="1",
        resource_manifest_digest="digest",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    return BatchAvailability(
        sample_interactions=pd.DataFrame.from_records(rows),
        mapping_summary=pd.DataFrame(),
        resource_id="fixture",
        resource_version="1",
        detection_available=True,
        frozen_interaction_universe=universe,
        filter_application=InteractionFilterApplication.TRAINING_SELECTION_V1,
        application_subject_ids=subjects,
    )


def _assignment(availability: BatchAvailability):
    table = availability.sample_interactions.loc[
        :,
        [
            "sample_id",
            "subject_id",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "ligand_availability",
        ],
    ]
    return assign_senders(table, SenderEvidenceParameters(min_subjects=3))


def test_absolute_heads_separate_detection_attribution_and_null_sender() -> None:
    availability = _availability()
    result = build_absolute_activity_heads(availability, _assignment(availability))

    assert tuple(result.columns) == ABSOLUTE_ACTIVITY_HEAD_COLUMNS
    assert set(result["score_version"]) == {ABSOLUTE_ACTIVITY_SCORE_VERSION}
    assert not result["formal_inference_allowed"].any()
    assert not any("probability" in column for column in result.columns)
    for _, group in result.groupby("sample_id", observed=True):
        strong = group.loc[group["sender"].eq("strong")].iloc[0]
        weak = group.loc[group["sender"].eq("weak")].iloc[0]
        assert strong["parent_activity_raw"] == 0.7
        assert strong["sender_detection"] == 0.7
        assert weak["sender_detection"] == 0.35
        assert group["sender_detection"].sum() > strong["parent_activity_raw"]
        assert np.isclose(group["sender_attribution"].sum(), 1.0)
        assert np.isclose(group["null_sender_attribution"].iloc[0], 0.2)
        assert np.isclose(
            group["sender_attribution_with_null"].sum()
            + group["null_sender_attribution"].iloc[0],
            1.0,
        )
        assert np.isclose(group["sender_mass"].sum(), strong["parent_activity_raw"])


def test_detection_is_invariant_to_an_irrelevant_candidate_sender() -> None:
    base = _availability()
    expanded = _availability(include_decoy=True)
    base_heads = build_absolute_activity_heads(base, _assignment(base))
    expanded_heads = build_absolute_activity_heads(expanded, _assignment(expanded))
    keys = ["sample_id", "sender", "receiver", "interaction_id"]
    shared = expanded_heads.loc[expanded_heads["sender"].isin(("strong", "weak"))]
    comparison = base_heads.merge(
        shared,
        on=keys,
        suffixes=("_base", "_expanded"),
        validate="one_to_one",
    )
    assert np.allclose(
        comparison["sender_detection_base"], comparison["sender_detection_expanded"]
    )
    assert np.allclose(
        comparison["parent_activity_raw_base"],
        comparison["parent_activity_raw_expanded"],
    )
    assert np.allclose(
        comparison["null_sender_attribution_base"],
        comparison["null_sender_attribution_expanded"],
    )
    assert (
        comparison["sender_attribution_expanded"]
        < comparison["sender_attribution_base"]
    ).all()


def test_low_absolute_sender_evidence_assigns_most_mass_to_null() -> None:
    availability = _availability(low_evidence=True)
    result = build_absolute_activity_heads(availability, _assignment(availability))

    for _, group in result.groupby("sample_id", observed=True):
        assert group["null_sender_attribution"].iloc[0] == 0.95
        assert np.isclose(group["sender_attribution_with_null"].sum(), 0.05)


def test_detection_remains_available_without_conditional_attribution() -> None:
    result = build_absolute_activity_heads(_availability(), None)

    assert result["sender_detection"].notna().all()
    assert result["sender_attribution"].isna().all()
    assert set(result["attribution_status"]) == {"not_estimable"}
    assert set(result["attribution_reason_code"]) == {"sender_assignment_not_supplied"}


def test_sample_missing_a_frozen_sender_keeps_detection_but_not_attribution() -> None:
    complete = _availability()
    assignment = _assignment(complete)
    source = complete.sample_interactions
    incomplete_table = source.loc[
        ~(source["sample_id"].eq("sample-0") & source["sender"].eq("weak"))
    ].copy()
    incomplete = BatchAvailability(
        sample_interactions=incomplete_table,
        mapping_summary=complete.mapping_summary,
        resource_id=complete.resource_id,
        resource_version=complete.resource_version,
        detection_available=complete.detection_available,
        frozen_interaction_universe=complete.frozen_interaction_universe,
        filter_application=complete.filter_application,
        application_subject_ids=complete.application_subject_ids,
    )

    result = build_absolute_activity_heads(incomplete, assignment)
    missing_sample = result.loc[result["sample_id"].eq("sample-0")]
    complete_samples = result.loc[~result["sample_id"].eq("sample-0")]

    assert missing_sample["sender_detection"].notna().all()
    assert missing_sample["sender_attribution"].isna().all()
    assert set(missing_sample["attribution_status"]) == {"not_estimable"}
    assert set(missing_sample["attribution_reason_code"]) == {
        "sender_candidate_coverage_incomplete"
    }
    sums = complete_samples.groupby("sample_id", observed=True)[
        "sender_attribution"
    ].sum()
    assert np.allclose(sums, 1.0)
