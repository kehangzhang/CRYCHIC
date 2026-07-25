from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.core import stable_id
from crychic.design import balanced_contrast, node_context_fields
from crychic.sender import (
    EBShrunkenCouplingStatus,
    EBShrunkenCouplingV2Spec,
    build_coupling_subject_effects_v2,
    coupling_contrast_weights_v2,
    fit_eb_shrunken_coupling_v2,
)

CONTEXT_KEYS = ("condition",)
CONTRAST = balanced_contrast(
    ("stim",),
    ("control",),
    name="stim_vs_control",
)


def _tables(*, paired: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activity_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    subjects = tuple(f"subject-{index:02d}" for index in range(8))
    for subject_index, subject in enumerate(subjects):
        contexts = (
            ("control", "stim")
            if paired
            else (("control",) if subject_index < 4 else ("stim",))
        )
        receiver_shift = 2.0 * (subject_index + 1)
        for condition in contexts:
            sample_id = f"sample-{subject}-{condition}"
            context_id = node_context_fields(condition, CONTEXT_KEYS)[0]
            metadata_rows.append(
                {
                    "sample": sample_id,
                    "patient": subject,
                    "condition": condition,
                    "age": 40.0 + subject_index,
                    "cohort": "site-a" if subject_index < 4 else "site-b",
                }
            )
            program_rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject,
                    "receiver": "receiver",
                    "interaction_id": "L_R",
                    "program_signed": (
                        5.0 + receiver_shift if condition == "stim" else 5.0
                    ),
                    "program_status": "observed",
                }
            )
            for sender, shift in (
                ("sender-a", float(subject_index + 1)),
                ("sender-b", float(np.sin(subject_index * 1.3))),
            ):
                activity_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject,
                        "context_id": context_id,
                        "sender": sender,
                        "receiver": "receiver",
                        "interaction_id": "L_R",
                        "ligand_activity_raw": (
                            3.0 + shift if condition == "stim" else 3.0
                        ),
                    }
                )
    return (
        pd.DataFrame(activity_rows),
        pd.DataFrame(program_rows),
        pd.DataFrame(metadata_rows),
    )


def _build(*, paired: bool) -> pd.DataFrame:
    activity, program, metadata = _tables(paired=paired)
    return build_coupling_subject_effects_v2(
        activity,
        program,
        metadata,
        contrast=CONTRAST,
        context_keys=CONTEXT_KEYS,
        training_subject_ids=tuple(f"subject-{index:02d}" for index in range(8)),
        sample_key="sample",
        subject_key="patient",
        numerical_covariates=("age",),
        categorical_covariates=("cohort",),
    )


def test_builder_emits_exact_paired_subject_contrasts_and_covariates() -> None:
    result = _build(paired=True)

    assert len(result) == 16
    assert not result[["sender_effect", "receiver_effect"]].isna().any().any()
    sender_a = result.loc[result["sender"].eq("sender-a")].reset_index(drop=True)
    assert sender_a["sender_effect"].to_numpy() == pytest.approx(np.arange(1.0, 9.0))
    assert sender_a["receiver_effect"].to_numpy() == pytest.approx(
        2.0 * np.arange(1.0, 9.0)
    )
    assert sender_a["age"].to_numpy() == pytest.approx(np.arange(40.0, 48.0))
    assert set(sender_a["cohort"]) == {"site-a", "site-b"}

    weights = coupling_contrast_weights_v2(CONTRAST, CONTEXT_KEYS)
    functional = fit_eb_shrunken_coupling_v2(
        result,
        fold_id="fold-1",
        training_subject_ids=tuple(f"subject-{index:02d}" for index in range(8)),
        training_input_digest="training-input-digest",
        sender_activity_transform_id="activity-transform",
        receiver_program_functional_id="program-functional",
        spec=EBShrunkenCouplingV2Spec(
            contrast_name=CONTRAST.name,
            minimum_subjects=4,
            minimum_observed_edges_for_eb=2,
        ),
        contrast_id=stable_id("contrast", CONTRAST.to_dict()),
        contrast_name=CONTRAST.name,
        contrast_weights=weights,
    )

    assert functional.status is EBShrunkenCouplingStatus.OBSERVED
    assert functional.contrast_name == CONTRAST.name
    assert functional.contrast_weights == weights


def test_independent_subjects_remain_explicitly_not_estimable() -> None:
    result = _build(paired=False)

    assert len(result) == 16
    assert result["sender_effect"].isna().all()
    assert result["receiver_effect"].isna().all()
    functional = fit_eb_shrunken_coupling_v2(
        result,
        fold_id="fold-1",
        training_subject_ids=tuple(f"subject-{index:02d}" for index in range(8)),
        training_input_digest="training-input-digest",
        sender_activity_transform_id="activity-transform",
        receiver_program_functional_id="program-functional",
        spec=EBShrunkenCouplingV2Spec(
            minimum_subjects=4,
            minimum_observed_edges_for_eb=2,
        ),
    )

    assert functional.status is EBShrunkenCouplingStatus.NOT_ESTIMABLE
    assert {record.reason_code for record in functional.records} == {
        "insufficient_complete_training_subjects"
    }


def test_coupling_fit_rejects_nonrectangular_edge_subject_input() -> None:
    result = _build(paired=True)
    incomplete = result.drop(index=result.index[0]).reset_index(drop=True)

    with pytest.raises(ValueError, match="explicitly cover every training subject"):
        fit_eb_shrunken_coupling_v2(
            incomplete,
            fold_id="fold-1",
            training_subject_ids=tuple(f"subject-{index:02d}" for index in range(8)),
            training_input_digest="training-input-digest",
            sender_activity_transform_id="activity-transform",
            receiver_program_functional_id="program-functional",
            spec=EBShrunkenCouplingV2Spec(
                minimum_subjects=4,
                minimum_observed_edges_for_eb=2,
            ),
        )
