from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from crychic.sender import (
    EB_SHRUNKEN_COUPLING_V2_VERSION,
    EBShrunkenCouplingStatus,
    EBShrunkenCouplingV2Spec,
    fit_eb_shrunken_coupling_v2,
)


def _effects(n_subjects: int = 20) -> pd.DataFrame:
    index = np.arange(n_subjects, dtype=float)
    receiver = np.sin(index * 0.73) + 0.1 * np.cos(index * 0.19)
    candidate_values = {
        "true": receiver + 0.08 * np.cos(index * 1.31),
        "weak": 0.35 * receiver + np.cos(index * 1.11),
        "null": np.cos(index * 0.47),
        "inverse": -receiver + 0.1 * np.sin(index * 1.51),
    }
    rows = [
        {
            "subject_id": f"subject-{subject_index:02d}",
            "sender": sender,
            "receiver": "receiver",
            "interaction_id": "L_R",
            "sender_effect": float(values[subject_index]),
            "receiver_effect": float(receiver[subject_index]),
        }
        for sender, values in candidate_values.items()
        for subject_index in range(n_subjects)
    ]
    return pd.DataFrame(rows)


def _fit(table: pd.DataFrame, spec: EBShrunkenCouplingV2Spec | None = None):
    return fit_eb_shrunken_coupling_v2(
        table,
        fold_id="fold-1",
        training_subject_ids=tuple(sorted(table["subject_id"].unique())),
        training_input_digest="training-input-digest",
        sender_activity_transform_id="activity-transform-id",
        receiver_program_functional_id="program-functional-id",
        spec=spec,
    )


def test_fisher_z_empirical_bayes_formula_is_exact() -> None:
    spec = EBShrunkenCouplingV2Spec(
        minimum_subjects=8,
        minimum_observed_edges_for_eb=2,
    )
    result = _fit(_effects(), spec)

    assert result.status is EBShrunkenCouplingStatus.OBSERVED
    assert result.tau2 is not None
    expected_components = []
    for record in result.records:
        assert record.status is EBShrunkenCouplingStatus.OBSERVED
        assert record.raw_correlation is not None
        assert record.fisher_z is not None
        assert record.sampling_variance == pytest.approx(1.0 / 17.0)
        expected_components.append(record.fisher_z**2 - record.sampling_variance)
        expected_factor = result.tau2 / (result.tau2 + record.sampling_variance)
        assert record.shrinkage_factor == pytest.approx(expected_factor)
        assert record.shrunken_correlation == pytest.approx(
            math.tanh(expected_factor * record.fisher_z)
        )
        assert abs(record.shrunken_correlation) <= abs(record.raw_correlation)
    assert result.tau2 == pytest.approx(max(0.0, np.mean(expected_components)))
    assert result.spec.to_dict()["version"] == EB_SHRUNKEN_COUPLING_V2_VERSION


def test_true_sender_remains_top_after_shrinkage_and_sign_is_preserved() -> None:
    result = _fit(
        _effects(),
        EBShrunkenCouplingV2Spec(
            minimum_subjects=8,
            minimum_observed_edges_for_eb=2,
        ),
    )
    by_sender = {item.sender: item for item in result.records}

    assert by_sender["true"].shrunken_correlation is not None
    assert by_sender["weak"].shrunken_correlation is not None
    assert (
        by_sender["true"].shrunken_correlation > by_sender["weak"].shrunken_correlation
    )
    assert by_sender["inverse"].shrunken_correlation < 0.0


def test_recorded_confounding_is_removed_before_eb_shrinkage() -> None:
    n_subjects = 24
    index = np.arange(n_subjects, dtype=float)
    batch = np.where(index % 2 == 0, "a", "b")
    batch_value = (batch == "b").astype(float)
    receiver = 5.0 * batch_value + np.sin(index * 0.7)
    rows = []
    for sender, values in (
        ("confounded", 8.0 * batch_value + np.cos(index * 0.3)),
        ("coupled", 2.0 * receiver + np.cos(index * 1.3) * 0.05),
    ):
        rows.extend(
            {
                "subject_id": f"subject-{subject_index:02d}",
                "sender": sender,
                "receiver": "receiver",
                "interaction_id": "L_R",
                "sender_effect": float(values[subject_index]),
                "receiver_effect": float(receiver[subject_index]),
                "batch": str(batch[subject_index]),
            }
            for subject_index in range(n_subjects)
        )
    result = _fit(
        pd.DataFrame(rows),
        EBShrunkenCouplingV2Spec(
            categorical_covariates=("batch",),
            minimum_subjects=8,
            minimum_observed_edges_for_eb=2,
        ),
    )
    by_sender = {item.sender: item for item in result.records}

    assert by_sender["coupled"].shrunken_correlation is not None
    assert by_sender["confounded"].shrunken_correlation is not None
    assert by_sender["coupled"].shrunken_correlation > 0.9
    assert abs(by_sender["confounded"].shrunken_correlation) < 0.5


def test_too_few_observed_edges_returns_typed_unavailable_functional() -> None:
    one_edge = _effects().loc[lambda table: table["sender"].eq("true")]
    result = _fit(
        one_edge,
        EBShrunkenCouplingV2Spec(
            minimum_subjects=8,
            minimum_observed_edges_for_eb=2,
        ),
    )

    assert result.status is EBShrunkenCouplingStatus.NOT_ESTIMABLE
    assert result.reason_code == "insufficient_observed_edges_for_eb"
    assert result.tau2 is None
    assert result.records[0].raw_correlation is not None
    assert result.records[0].shrunken_correlation is None


def test_coupling_v2_is_deterministic_and_never_emits_formal_inference() -> None:
    table = _effects()
    spec = EBShrunkenCouplingV2Spec(
        minimum_subjects=8,
        minimum_observed_edges_for_eb=2,
    )
    first = _fit(table, spec)
    second = _fit(table.sample(frac=1.0, random_state=13), spec)

    assert first.functional_id == second.functional_id
    assert first.to_dict() == second.to_dict()
    assert not first.formal_inference_allowed
    assert "p_value" not in first.to_dict()
    assert "q_value" not in first.to_dict()


def test_coupling_v2_rejects_incomplete_training_subject_lineage() -> None:
    table = _effects()
    subjects = tuple(sorted(table["subject_id"].unique()))

    with pytest.raises(ValueError, match="exactly cover training_subject_ids"):
        fit_eb_shrunken_coupling_v2(
            table,
            fold_id="fold-1",
            training_subject_ids=subjects[:-1],
            training_input_digest="training-input-digest",
            sender_activity_transform_id="activity-transform-id",
            receiver_program_functional_id="program-functional-id",
            spec=EBShrunkenCouplingV2Spec(minimum_observed_edges_for_eb=2),
        )
