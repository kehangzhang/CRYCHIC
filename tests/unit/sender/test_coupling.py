from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.sender import (
    RESIDUALIZED_COUPLING_VERSION,
    ResidualizedCouplingSpec,
    ResidualizedCouplingStatus,
    fit_residualized_sender_coupling,
)


def _fit(table: pd.DataFrame, spec: ResidualizedCouplingSpec):
    return fit_residualized_sender_coupling(
        table,
        sender="Sender",
        receiver="Receiver",
        interaction_id="L_R",
        fold_id="fold-1",
        spec=spec,
    )


def test_residualized_coupling_removes_batch_and_retains_shared_signal() -> None:
    index = np.arange(24, dtype=float)
    batch = np.where(index % 2 == 0, "a", "b")
    batch_value = (batch == "b").astype(float)
    signal = np.sin(index * 0.7) + index / 40.0
    table = pd.DataFrame(
        {
            "subject_id": [f"s{value:02.0f}" for value in index],
            "sender_effect": signal + 4.0 * batch_value,
            "receiver_effect": 2.0 * signal + 7.0 * batch_value,
            "batch": batch,
        }
    )
    result = _fit(
        table,
        ResidualizedCouplingSpec(categorical_covariates=("batch",), minimum_subjects=8),
    )

    assert result.status is ResidualizedCouplingStatus.OBSERVED
    assert result.signed_correlation == pytest.approx(1.0)
    assert result.positive_coupling_support == pytest.approx(1.0)
    assert result.design_columns == ("intercept", "categorical:batch=b")
    assert result.algorithm_version == RESIDUALIZED_COUPLING_VERSION
    assert result.formal_inference_allowed is False
    assert set(result.to_dict()["excluded_output_kinds"]) == {
        "p_value",
        "q_value",
        "posterior_probability",
        "communication_probability",
    }


def test_pure_confounding_becomes_typed_not_estimable_not_zero_support() -> None:
    batch = np.asarray(["a", "b"] * 6)
    batch_value = (batch == "b").astype(float)
    table = pd.DataFrame(
        {
            "subject_id": [f"s{index:02d}" for index in range(len(batch))],
            "sender_effect": batch_value,
            "receiver_effect": 3.0 * batch_value,
            "batch": batch,
        }
    )
    result = _fit(
        table,
        ResidualizedCouplingSpec(categorical_covariates=("batch",), minimum_subjects=8),
    )

    assert result.status is ResidualizedCouplingStatus.NOT_ESTIMABLE
    assert result.reason_code == "residual_variance_zero"
    assert result.signed_correlation is None
    assert result.positive_coupling_support is None


def test_numerical_composition_residualization_suppresses_decoy_coupling() -> None:
    composition = np.linspace(-1.0, 1.0, 20)
    orthogonal = np.tile(np.asarray([-1.0, 1.0]), 10)
    table = pd.DataFrame(
        {
            "subject_id": [f"s{index:02d}" for index in range(20)],
            "sender_effect": 2.0 * composition + orthogonal,
            "receiver_effect": 3.0 * composition - orthogonal,
            "sender_proportion_effect": composition,
        }
    )
    raw = _fit(table, ResidualizedCouplingSpec(minimum_subjects=8))
    adjusted = _fit(
        table,
        ResidualizedCouplingSpec(
            numerical_covariates=("sender_proportion_effect",),
            minimum_subjects=8,
        ),
    )

    assert raw.signed_correlation is not None and raw.signed_correlation > 0.3
    assert adjusted.signed_correlation == pytest.approx(-1.0)
    assert adjusted.positive_coupling_support == 0.0


def test_coupling_is_deterministic_under_subject_order() -> None:
    index = np.arange(12, dtype=float)
    table = pd.DataFrame(
        {
            "subject_id": [f"s{value:02.0f}" for value in index],
            "sender_effect": np.sin(index),
            "receiver_effect": np.sin(index) + 0.1 * np.cos(index),
        }
    )
    spec = ResidualizedCouplingSpec(minimum_subjects=8)
    first = _fit(table, spec)
    second = _fit(table.sample(frac=1.0, random_state=7), spec)

    assert first.coupling_id == second.coupling_id
    assert first.to_dict() == second.to_dict()


def test_missing_subjects_and_duplicate_subjects_fail_or_degrade_explicitly() -> None:
    table = pd.DataFrame(
        {
            "subject_id": [f"s{index:02d}" for index in range(8)],
            "sender_effect": np.arange(8, dtype=float),
            "receiver_effect": np.arange(8, dtype=float),
        }
    )
    result = _fit(table.iloc[:6], ResidualizedCouplingSpec(minimum_subjects=8))
    assert result.status is ResidualizedCouplingStatus.NOT_ESTIMABLE
    assert result.reason_code == "insufficient_complete_training_subjects"

    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per subject"):
        _fit(duplicate, ResidualizedCouplingSpec(minimum_subjects=8))
