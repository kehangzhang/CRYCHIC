from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.design import ContrastSpec, balanced_contrast
from crychic.sender import (
    CommonSenderApplication,
    CommonSenderApplicationStatus,
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    SenderPrevalenceStatus,
    allocate_sender_resolved_strength,
    apply_contrast_common_sender_functional,
    fit_contrast_common_sender_functional,
)


def _training_availability() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        ("p1", "control"): {"A": 0.8, "B": 0.0},
        ("p1", "treated"): {"A": 0.6, "B": 0.2},
        ("p2", "control"): {"A": 0.4, "B": 0.0},
        ("p2", "treated"): {"A": 0.2, "B": 0.0},
    }
    for (subject, context), candidates in values.items():
        for sender, ligand in candidates.items():
            rows.append(
                {
                    "sample_id": f"{subject}-{context}",
                    "subject_id": subject,
                    "context_id": context,
                    "sender": sender,
                    "receiver": "R",
                    "interaction_id": "L_R",
                    "ligand_availability": ligand,
                }
            )
    return pd.DataFrame(rows)


def _heldout_availability(*, context: str = "treated") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["test-1", "test-1"],
            "subject_id": ["test-subject", "test-subject"],
            "context_id": [context, context],
            "sender": ["A", "B"],
            "receiver": ["R", "R"],
            "interaction_id": ["L_R", "L_R"],
            "ligand_availability": [0.25, 0.75],
        }
    )


def _functional() -> ContrastCommonSenderFunctional:
    return fit_contrast_common_sender_functional(
        _training_availability(),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        parameters=ContrastCommonSenderParameters(
            min_subjects=2,
            prevalence_threshold=0.0,
            softmax_temperature=0.5,
        ),
    )


def _contrast() -> ContrastSpec:
    return balanced_contrast(
        ("treated",),
        ("control",),
        name="treated-v-control",
        family="test",
    )


def test_fit_pools_prevalence_across_contexts_and_is_order_stable() -> None:
    source = _training_availability()
    first = _functional()
    second = fit_contrast_common_sender_functional(
        source.sample(frac=1.0, random_state=17),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        parameters=first.parameters,
    )

    assert first.sender_functional_id == second.sender_functional_id
    assert first.context_ids == ("control", "treated")
    assert first.training_subject_ids == ("p1", "p2")
    assert first.to_dict()["common_across_contexts"] is True
    assert first.to_dict()["causal_interpretation"] == ("evidence_based_non_causal")
    priors = {prior.sender: prior for prior in first.candidate_priors}
    assert priors["A"].prevalence_prior == pytest.approx(1.0)
    assert priors["B"].prevalence_prior == pytest.approx(0.5)
    assert {prior.status for prior in priors.values()} == {
        SenderPrevalenceStatus.SUPPORTED
    }


def test_unrelated_third_context_does_not_change_pairwise_functional_id() -> None:
    base = _training_availability()
    third = base.loc[base["context_id"].eq("control")].copy(deep=True)
    third["context_id"] = "unrelated"
    third["sample_id"] = third["sample_id"].str.replace(
        "control", "unrelated", regex=False
    )
    expanded = pd.concat([base, third], ignore_index=True)

    pairwise = fit_contrast_common_sender_functional(
        expanded,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        parameters=_functional().parameters,
    )

    assert pairwise.sender_functional_id == _functional().sender_functional_id
    assert pairwise.context_ids == ("control", "treated")


def test_missing_contrast_context_is_rejected_instead_of_single_context_fit() -> None:
    single = _training_availability().loc[
        lambda frame: frame["context_id"].eq("treated")
    ]

    with pytest.raises(ValueError, match="absent from training"):
        fit_contrast_common_sender_functional(
            single,
            contrast=_contrast(),
            context_keys=("context_id",),
            filter_universe_id="filter-universe-1",
            parameters=ContrastCommonSenderParameters(min_subjects=2),
        )


def test_application_uses_only_local_ligand_and_one_frozen_functional() -> None:
    functional = _functional()
    result = apply_contrast_common_sender_functional(
        functional, _heldout_availability()
    )
    table = result.table.set_index("sender")

    raw_a = 0.25 * 1.0
    raw_b = 0.75 * 0.5
    expected_a = math.exp(raw_a / 0.5) / (math.exp(raw_a / 0.5) + math.exp(raw_b / 0.5))
    assert table.loc["A", "raw_sender_evidence"] == pytest.approx(raw_a)
    assert table.loc["B", "raw_sender_evidence"] == pytest.approx(raw_b)
    assert table.loc["A", "assignment_weight"] == pytest.approx(expected_a)
    assert table["assignment_weight"].sum() == pytest.approx(1.0)
    assert set(table["sender_functional_id"]) == {functional.sender_functional_id}
    assert set(table["status"]) == {CommonSenderApplicationStatus.OK.value}
    assert result.causal_interpretation == "evidence_based_non_causal"
    assert not result.is_oof_certified


def test_unrelated_heldout_context_is_excluded_and_unseen_sender_is_ignored() -> None:
    functional = _functional()
    original = apply_contrast_common_sender_functional(
        functional, _heldout_availability(context="treated")
    )
    relabeled = apply_contrast_common_sender_functional(
        functional, _heldout_availability(context="new-heldout-context")
    )
    poisoned = pd.concat(
        [
            _heldout_availability(),
            pd.DataFrame(
                [
                    {
                        "sample_id": "test-1",
                        "subject_id": "test-subject",
                        "context_id": "treated",
                        "sender": "test-only-poison",
                        "receiver": "R",
                        "interaction_id": "L_R",
                        "ligand_availability": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with_poison = apply_contrast_common_sender_functional(functional, poisoned)

    assert relabeled.table.empty
    pd.testing.assert_series_equal(
        original.table.set_index("sender")["assignment_weight"],
        with_poison.table.set_index("sender")["assignment_weight"],
    )
    assert "test-only-poison" not in set(with_poison.table["sender"])


def test_all_missing_local_evidence_stays_na_without_uniform_fallback() -> None:
    heldout = _heldout_availability()
    heldout["ligand_availability"] = np.nan

    result = apply_contrast_common_sender_functional(_functional(), heldout)

    assert result.table["assignment_weight"].isna().all()
    assert result.table["normalized_entropy"].isna().all()
    assert set(result.table["status"]) == {
        CommonSenderApplicationStatus.NOT_ESTIMABLE.value
    }
    assert set(result.table["reason_code"]) == {"all_candidate_evidence_missing"}


def test_training_prior_and_parameter_poison_change_functional_identity() -> None:
    source = _training_availability()
    first = _functional()
    changed = source.copy(deep=True)
    changed.loc[changed["sender"].eq("B"), "ligand_availability"] = 0.5
    changed_prior = fit_contrast_common_sender_functional(
        changed,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        parameters=first.parameters,
    )
    changed_temperature = fit_contrast_common_sender_functional(
        source,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        parameters=ContrastCommonSenderParameters(
            min_subjects=2,
            prevalence_threshold=0.0,
            softmax_temperature=0.75,
        ),
    )

    assert first.sender_functional_id != changed_prior.sender_functional_id
    assert first.sender_functional_id != changed_temperature.sender_functional_id


def test_common_sender_parameters_reject_forced_mutation_before_use() -> None:
    parameters = ContrastCommonSenderParameters(min_subjects=2)
    object.__setattr__(parameters, "min_subjects", 999)

    with pytest.raises(ContractError) as serialized_error:
        parameters.to_dict()
    assert (
        serialized_error.value.details.code
        == "common_sender_parameter_integrity_violation"
    )

    with pytest.raises(ContractError) as fit_error:
        fit_contrast_common_sender_functional(
            _training_availability(),
            contrast=_contrast(),
            context_keys=("context_id",),
            filter_universe_id="filter-universe-1",
            parameters=parameters,
        )
    assert fit_error.value.details.code == (
        "common_sender_parameter_integrity_violation"
    )


def test_low_support_prior_cannot_be_fabricated_at_application() -> None:
    source = _training_availability()
    source.loc[source["sender"].eq("B"), "ligand_availability"] = np.nan
    functional = fit_contrast_common_sender_functional(
        source,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        parameters=ContrastCommonSenderParameters(min_subjects=2),
    )
    application = apply_contrast_common_sender_functional(
        functional, _heldout_availability()
    )
    result: pd.DataFrame = application.table.set_index("sender")

    prior_value: Any = result.loc["B", "training_prevalence_prior"]
    assert pd.isna(prior_value)
    assert result["assignment_weight"].isna().all()
    assert set(result["status"]) == {
        CommonSenderApplicationStatus.NOT_ESTIMABLE.value
    }
    assert set(result["reason_code"]) == {"incomplete_candidate_evidence"}


def test_sender_resolution_exactly_conserves_unresolved_strength() -> None:
    application = apply_contrast_common_sender_functional(
        _functional(), _heldout_availability()
    )
    unresolved = pd.DataFrame(
        [
            {
                "sample_id": "test-1",
                "subject_id": "test-subject",
                "context_id": "treated",
                "receiver": "R",
                "interaction_id": "L_R",
                "mode": "state",
                "sender_unresolved_strength": 0.8,
            }
        ]
    )

    resolved = allocate_sender_resolved_strength(application, unresolved)

    assert resolved["sender_unresolved_strength"].nunique() == 1
    assert resolved["sender_unresolved_strength"].iloc[0] == pytest.approx(0.8)
    assert resolved["sender_resolved_strength"].sum() == pytest.approx(0.8)
    assert set(resolved["status"]) == {"ok"}


def test_common_sender_rejects_receiver_outcomes_and_does_not_mutate_inputs() -> None:
    training = _training_availability()
    heldout = _heldout_availability()
    training_before = training.copy(deep=True)
    heldout_before = heldout.copy(deep=True)

    with pytest.raises(ContractError, match="must not consume receiver outcomes"):
        fit_contrast_common_sender_functional(
            training.assign(receiver_activity=1.0),
            contrast=_contrast(),
            context_keys=("context_id",),
            filter_universe_id="filter-universe-1",
        )
    functional = _functional()
    apply_contrast_common_sender_functional(functional, heldout)

    pd.testing.assert_frame_equal(training, training_before)
    pd.testing.assert_frame_equal(heldout, heldout_before)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "column",
    [
        "sender_application_id",
        "training_prevalence_prior",
        "raw_sender_evidence",
        "assignment_weight",
        "status",
        "reason_code",
    ],
)
def test_application_contract_rejects_frozen_functional_poison(column: str) -> None:
    application = apply_contrast_common_sender_functional(
        _functional(), _heldout_availability()
    )
    poisoned = application.table.copy(deep=True)
    if column == "sender_application_id":
        poisoned.loc[0, column] = "common_sender_application_poison"
    elif column == "status":
        poisoned.loc[0, column] = CommonSenderApplicationStatus.NOT_ESTIMABLE.value
        poisoned.loc[0, "reason_code"] = "arbitrary_reason"
    elif column == "reason_code":
        poisoned.loc[0, column] = "arbitrary_reason"
    else:
        poisoned.loc[0, column] = 0.0

    with pytest.raises(ContractError):
        CommonSenderApplication(poisoned, application.functional)


def test_application_contract_rejects_context_and_candidate_omission() -> None:
    application = apply_contrast_common_sender_functional(
        _functional(), _heldout_availability()
    )
    wrong_context = application.table.copy(deep=True)
    wrong_context["context_id"] = "unrelated"
    omitted_candidate = application.table.loc[
        application.table["sender"].eq("A")
    ].copy(deep=True)
    omitted_candidate["assignment_weight"] = 1.0
    omitted_candidate["normalized_entropy"] = 0.0

    with pytest.raises(ContractError, match="outside the bound contrast"):
        CommonSenderApplication(wrong_context, application.functional)
    with pytest.raises(ContractError, match="frozen sender universe"):
        CommonSenderApplication(omitted_candidate, application.functional)
