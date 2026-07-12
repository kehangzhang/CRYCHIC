from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.sender import (
    SENDER_ASSIGNMENT_COLUMNS,
    SenderAssignment,
    SenderAssignmentStatus,
    SenderEvidenceParameters,
    assign_senders,
)


def _availability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "subject_id": ["p1", "p1", "p2", "p2"],
            "context_id": ["treated"] * 4,
            "sender": ["A", "B", "A", "B"],
            "receiver": ["R"] * 4,
            "interaction_id": ["L_R"] * 4,
            "ligand_availability": [0.8, 0.2, 0.4, 0.0],
        }
    )


def test_hand_calculated_components_softmax_and_entropy() -> None:
    parameters = SenderEvidenceParameters(min_subjects=2)

    result = assign_senders(_availability(), parameters)
    table = result.table.set_index("sender")

    ligand_a, ligand_b = 0.6, 0.1
    specificity_a, specificity_b = 6 / 7, 1 / 7
    prevalence_a, prevalence_b = 1.0, 0.5
    score_a = (ligand_a + specificity_a + prevalence_a) / 3
    score_b = (ligand_b + specificity_b + prevalence_b) / 3
    weight_a = math.exp(score_a) / (math.exp(score_a) + math.exp(score_b))
    weight_b = 1 - weight_a
    entropy = -(
        weight_a * math.log(weight_a) + weight_b * math.log(weight_b)
    ) / math.log(2)

    assert table.loc["A", "ligand_availability"] == pytest.approx(ligand_a)
    assert table.loc["B", "ligand_availability"] == pytest.approx(ligand_b)
    assert table.loc["A", "cell_type_specificity"] == pytest.approx(specificity_a)
    assert table.loc["B", "cell_type_specificity"] == pytest.approx(specificity_b)
    assert table.loc["A", "subject_prevalence"] == pytest.approx(prevalence_a)
    assert table.loc["B", "subject_prevalence"] == pytest.approx(prevalence_b)
    assert table.loc["A", "evidence_score"] == pytest.approx(score_a)
    assert table.loc["B", "evidence_score"] == pytest.approx(score_b)
    assert table.loc["A", "assignment_weight"] == pytest.approx(weight_a)
    assert table.loc["B", "assignment_weight"] == pytest.approx(weight_b)
    assert table["assignment_weight"].sum() == pytest.approx(1.0)
    assert table["normalized_entropy"].tolist() == pytest.approx([entropy, entropy])
    assert set(table["n_subjects"]) == {2}
    assert set(table["n_group_subjects"]) == {2}
    assert set(table["status"]) == {SenderAssignmentStatus.OK.value}
    assert result.causal_interpretation == "evidence_based_non_causal"
    assert not result.inference_eligible


def test_indistinguishable_senders_have_equal_weights_and_maximum_entropy() -> None:
    rows = []
    for subject in ("p1", "p2", "p3"):
        for sender in ("A", "B", "C"):
            rows.append(
                {
                    "sample_id": f"sample-{subject}",
                    "subject_id": subject,
                    "context_id": "control",
                    "sender": sender,
                    "receiver": "R",
                    "interaction_id": "L_R",
                    "ligand_availability": 0.5,
                }
            )

    result = assign_senders(pd.DataFrame(rows))

    assert result.table["assignment_weight"].tolist() == pytest.approx([1 / 3] * 3)
    assert result.table["normalized_entropy"].tolist() == pytest.approx([1.0] * 3)
    assert result.table["cell_type_specificity"].tolist() == pytest.approx([1 / 3] * 3)


def test_context_batch_covariation_cannot_create_v0_1_coupling() -> None:
    rows = []
    for context, batch, activity, subjects in (
        ("control", "batch-0", 0.0, ("p1", "p2")),
        ("treated", "batch-1", 1.0, ("p3", "p4")),
    ):
        for subject in subjects:
            for sender, ligand in (("A", 0.8), ("B", 0.2)):
                rows.append(
                    {
                        "sample_id": f"{subject}-{context}",
                        "subject_id": subject,
                        "context_id": context,
                        "sender": sender,
                        "receiver": "R",
                        "interaction_id": "L_R",
                        "ligand_availability": ligand,
                        "batch": batch,
                        "receiver_activity": activity,
                    }
                )
    confounded = pd.DataFrame(rows)

    with_extras = assign_senders(confounded, SenderEvidenceParameters(min_subjects=2))
    availability_only = assign_senders(
        confounded.loc[
            :, sorted(set(confounded.columns) - {"batch", "receiver_activity"})
        ],
        SenderEvidenceParameters(min_subjects=2),
    )

    pd.testing.assert_frame_equal(with_extras.table, availability_only.table)
    assert with_extras.table["adjusted_coupling"].isna().all()
    assert (with_extras.table["coupling_weight"] == 0).all()
    assert set(with_extras.table["coupling_status"]) == {"not_estimable_v0_1"}
    assert set(with_extras.table["coupling_reason_code"]) == {
        "v0_1_adjusted_coupling_disabled"
    }
    assert not with_extras.coupling_estimable


def test_missing_candidate_is_explicit_and_not_converted_to_zero() -> None:
    availability = _availability()
    availability.loc[availability["sender"] == "B", "ligand_availability"] = np.nan

    result = assign_senders(availability, SenderEvidenceParameters(min_subjects=2))
    table = result.table.set_index("sender")

    assert table.loc["A", "assignment_weight"] == pytest.approx(1.0)
    assert table.loc["A", "status"] == SenderAssignmentStatus.PARTIAL_EVIDENCE.value
    assert table.loc["A", "reason_code"] == "partial_candidate_evidence"
    assert table.loc["B", "assignment_weight"] == pytest.approx(0.0)
    assert table.loc["B", "status"] == SenderAssignmentStatus.MISSING_EVIDENCE.value
    assert table.loc["B", "reason_code"] == "missing_ligand_availability"
    assert math.isnan(float(cast(Any, table.loc["B", "ligand_availability"])))
    assert math.isnan(float(cast(Any, table.loc["B", "subject_prevalence"])))
    assert math.isnan(float(cast(Any, table.loc["B", "evidence_score"])))
    assert table.loc["B", "n_subjects"] == 0
    assert table["assignment_weight"].sum() == pytest.approx(1.0)


def test_all_missing_candidates_use_explicit_noninformative_equal_fallback() -> None:
    availability = _availability()
    availability["ligand_availability"] = np.nan

    result = assign_senders(availability)

    assert result.table["assignment_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert result.table["normalized_entropy"].tolist() == pytest.approx([1.0, 1.0])
    assert set(result.table["status"]) == {
        SenderAssignmentStatus.MISSING_EVIDENCE.value
    }
    assert set(result.table["reason_code"]) == {"all_candidate_evidence_missing"}


@pytest.mark.parametrize("candidate_count", [5, 13, 19])
def test_equal_weight_entropy_stays_inside_unit_interval(
    candidate_count: int,
) -> None:
    availability = pd.DataFrame(
        {
            "sample_id": ["s1"] * candidate_count,
            "subject_id": ["p1"] * candidate_count,
            "context_id": ["treated"] * candidate_count,
            "sender": [f"sender-{index}" for index in range(candidate_count)],
            "receiver": ["R"] * candidate_count,
            "interaction_id": ["L_R"] * candidate_count,
            "ligand_availability": [np.nan] * candidate_count,
        }
    )

    result = assign_senders(availability)

    assert result.table["normalized_entropy"].tolist() == [1.0] * candidate_count


def test_low_subject_support_is_retained_with_reason() -> None:
    one_subject = _availability().loc[lambda frame: frame["subject_id"] == "p1"]

    result = assign_senders(one_subject, SenderEvidenceParameters(min_subjects=2))

    assert result.table["assignment_weight"].sum() == pytest.approx(1.0)
    assert set(result.table["status"]) == {SenderAssignmentStatus.LOW_SUPPORT.value}
    assert set(result.table["reason_code"]) == {"insufficient_subject_support"}
    assert set(result.table["n_subjects"]) == {1}
    assert result.table["evidence_score"].notna().all()


def test_assignment_is_deterministic_under_input_order_and_defensive_query() -> None:
    original = _availability()
    shuffled = original.sample(frac=1.0, random_state=17).reset_index(drop=True)

    first = assign_senders(original, SenderEvidenceParameters(min_subjects=2))
    second = assign_senders(shuffled, SenderEvidenceParameters(min_subjects=2))

    pd.testing.assert_frame_equal(first.table, second.table)
    selected = first.query(sender="A")
    selected.loc[:, "assignment_weight"] = 0.0
    assert (
        first.table.loc[first.table["sender"] == "A", "assignment_weight"].iloc[0] > 0
    )


def test_contract_forbids_attribution_inference_and_invalid_normalization() -> None:
    with pytest.raises(ContractError, match="must not consume attribution"):
        assign_senders(_availability().assign(coefficient=0.9))
    with pytest.raises(ContractError, match="coupling must have weight zero"):
        SenderEvidenceParameters(
            component_weights={
                "ligand_availability": 1.0,
                "cell_type_specificity": 1.0,
                "subject_prevalence": 1.0,
                "adjusted_coupling": 1.0,
            }
        )

    result = assign_senders(_availability(), SenderEvidenceParameters(min_subjects=2))
    with pytest.raises(ContractError, match="inferential fields"):
        SenderAssignment(result.table.assign(p_value=0.1), result.parameters)
    invalid = result.table.copy()
    invalid["assignment_weight"] = 0.9
    with pytest.raises(ContractError, match="sum to one"):
        SenderAssignment(invalid, result.parameters)


def test_empty_input_preserves_producer_schema_and_exploratory_semantics() -> None:
    result = assign_senders(_availability().iloc[0:0].copy())

    assert result.table.empty
    assert tuple(result.table.columns) == SENDER_ASSIGNMENT_COLUMNS
    assert result.parameters.to_dict()["assignment_mode"] == "exploratory_in_sample"
    assert not any(
        column in result.table.columns for column in ("p_value", "q_value", "posterior")
    )
