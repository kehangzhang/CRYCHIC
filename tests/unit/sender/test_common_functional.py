from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

import crychic.sender as sender_module
from crychic.availability import FrozenInteractionUniverse, InteractionFilterPolicy
from crychic.core import ContractError
from crychic.design import ContrastSpec, balanced_contrast
from crychic.sender import (
    CommonSenderApplication,
    CommonSenderApplicationStatus,
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    InteractionLigandContrastSupport,
    SenderContrastSupportStatus,
    SenderPrevalenceStatus,
    allocate_sender_resolved_strength,
    apply_contrast_common_sender_functional,
    freeze_common_sender_candidate_manifest,
    interaction_ligand_contrast_gate,
    interaction_ligand_contrast_gates,
)
from crychic.sender.contracts import _SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN


def fit_contrast_common_sender_functional(
    training_availability: pd.DataFrame,
    **kwargs: Any,
) -> ContrastCommonSenderFunctional:
    filter_label = str(kwargs.pop("filter_universe_id"))
    frozen_interaction_ids = tuple(kwargs.pop("frozen_interaction_ids"))
    subjects = tuple(sorted(set(training_availability["subject_id"].astype(str))))
    universe = FrozenInteractionUniverse(
        interaction_ids=frozen_interaction_ids,
        training_subject_ids=subjects,
        resource_id=f"sender-unit:{filter_label}",
        resource_version="1",
        resource_manifest_digest=f"sender-unit-manifest:{filter_label}",
        min_pooled_availability=0.0,
        max_interactions=None,
        selection_policy=InteractionFilterPolicy.POOLED_SUPPORT_V1,
    )
    kwargs.setdefault(
        "frozen_candidate_sender_manifest",
        freeze_common_sender_candidate_manifest(
            training_availability,
            frozen_interaction_ids=frozen_interaction_ids,
        ),
    )
    kwargs.setdefault("training_input_digest", "sender-unit-training-input")
    kwargs["frozen_interaction_universe"] = universe
    return sender_module.fit_contrast_common_sender_functional(
        training_availability,
        **kwargs,
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


def _frozen_ids(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(set(table["interaction_id"].astype(str))))


def _functional() -> ContrastCommonSenderFunctional:
    return fit_contrast_common_sender_functional(
        _training_availability(),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        frozen_interaction_ids=("L_R",),
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


def _three_context_contrast(*, reverse: bool = False) -> ContrastSpec:
    weights = {"c": 1.0, "a": -0.5, "b": -0.5}
    if reverse:
        weights = {node: -weight for node, weight in weights.items()}
    return ContrastSpec(
        name="c-v-a-b" if not reverse else "a-b-v-c",
        weights=weights,
        family="three-context-test",
        mode="balanced",
    )


def _three_context_availability() -> pd.DataFrame:
    context_ids = {"a": "z-context", "b": "a-context", "c": "m-context"}
    rows: list[dict[str, object]] = []
    for subject_index in range(4):
        subject = f"multi-{subject_index}"
        values = {
            "a": 0.15 + 0.01 * subject_index,
            "b": 0.25 + 0.02 * subject_index,
            "c": 0.70 + 0.03 * subject_index,
        }
        for node, maximum in values.items():
            for sender, ligand in (("A", maximum), ("B", maximum - 0.05)):
                rows.append(
                    {
                        "sample_id": f"{subject}-{node}-{sender}",
                        "subject_id": subject,
                        "condition": node,
                        "context_id": context_ids[node],
                        "sender": sender,
                        "receiver": "R",
                        "interaction_id": "L_R",
                        "ligand_availability": ligand,
                    }
                )
    return pd.DataFrame(rows)


def _fit_three_context(
    contrast: ContrastSpec | None = None,
    *,
    source: pd.DataFrame | None = None,
) -> ContrastCommonSenderFunctional:
    table = _three_context_availability() if source is None else source
    return fit_contrast_common_sender_functional(
        table,
        contrast=contrast or _three_context_contrast(),
        context_keys=("condition",),
        filter_universe_id="three-context-order-test",
        frozen_interaction_ids=("L_R",),
        parameters=ContrastCommonSenderParameters(min_subjects=2),
    )


def _paired_interaction_availability(
    effects: tuple[float, ...],
    *,
    missing_treated_subjects: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing = set(missing_treated_subjects)
    for index, effect in enumerate(effects, start=1):
        subject = f"s{index}"
        for context, maximum in (
            ("control", 0.3),
            ("treated", 0.3 + effect),
        ):
            if context == "treated" and subject in missing:
                continue
            for sender, ligand in (("A", maximum), ("B", maximum - 0.1)):
                rows.append(
                    {
                        "sample_id": f"{subject}-{context}-{sender}",
                        "subject_id": subject,
                        "context_id": context,
                        "sender": sender,
                        "receiver": "R",
                        "interaction_id": "L_R",
                        "ligand_availability": ligand,
                    }
                )
    return pd.DataFrame(rows)


def _fit_paired_interaction(
    effects: tuple[float, ...],
    *,
    min_subjects: int = 2,
    missing_treated_subjects: tuple[str, ...] = (),
    ligand_contrast_minimum_effect: float = 0.0,
) -> ContrastCommonSenderFunctional:
    return fit_contrast_common_sender_functional(
        _paired_interaction_availability(
            effects, missing_treated_subjects=missing_treated_subjects
        ),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="interaction-support-test",
        frozen_interaction_ids=("L_R",),
        parameters=ContrastCommonSenderParameters(
            min_subjects=min_subjects,
            ligand_contrast_minimum_effect=ligand_contrast_minimum_effect,
        ),
    )


def _multi_interaction_availability(
    interaction_effects: dict[str, tuple[float, ...]],
    *,
    missing_treated_subjects: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    missing = missing_treated_subjects or {}
    frames: list[pd.DataFrame] = []
    for interaction_id, effects in interaction_effects.items():
        frame = _paired_interaction_availability(
            effects,
            missing_treated_subjects=missing.get(interaction_id, ()),
        )
        frame["interaction_id"] = interaction_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fit_multi_interactions(
    interaction_effects: dict[str, tuple[float, ...]],
    *,
    min_subjects: int = 2,
    missing_treated_subjects: dict[str, tuple[str, ...]] | None = None,
    ligand_contrast_minimum_effect: float = 0.0,
) -> ContrastCommonSenderFunctional:
    return fit_contrast_common_sender_functional(
        _multi_interaction_availability(
            interaction_effects,
            missing_treated_subjects=missing_treated_subjects,
        ),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="multi-interaction-support-test",
        frozen_interaction_ids=tuple(sorted(interaction_effects)),
        parameters=ContrastCommonSenderParameters(
            min_subjects=min_subjects,
            ligand_contrast_minimum_effect=ligand_contrast_minimum_effect,
        ),
    )


def _rebuilt_support(
    support: InteractionLigandContrastSupport,
    **overrides: object,
) -> InteractionLigandContrastSupport:
    values: dict[str, object] = {
        "receiver": support.receiver,
        "interaction_id": support.interaction_id,
        "complete_subject_ids": support.complete_subject_ids,
        "subject_effects": support.subject_effects,
        "minimum_complete_subjects": support.minimum_complete_subjects,
        "contrast_weights": support.contrast_weights,
        "ligand_contrast_confidence_level": (support.ligand_contrast_confidence_level),
        "ligand_contrast_minimum_effect": (support.ligand_contrast_minimum_effect),
        "degrees_of_freedom": support.degrees_of_freedom,
        "mean_effect": support.mean_effect,
        "sample_standard_error": support.sample_standard_error,
        "critical_value": support.critical_value,
        "lower_confidence_bound": support.lower_confidence_bound,
        "raw_one_sided_p_value": support.raw_one_sided_p_value,
        "multiplicity_method": support.multiplicity_method,
        "multiplicity_scope": support.multiplicity_scope,
        "multiplicity_family_id": support.multiplicity_family_id,
        "multiplicity_family_size": support.multiplicity_family_size,
        "holm_rank": support.holm_rank,
        "holm_adjusted_p_value": support.holm_adjusted_p_value,
        "status": support.status,
        "reason_code": support.reason_code,
    }
    values.update(overrides)
    return InteractionLigandContrastSupport._from_training(
        _producer_token=_SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN,
        **values,  # type: ignore[arg-type]
    )


def test_fit_pools_prevalence_across_contexts_and_is_order_stable() -> None:
    source = _training_availability()
    first = _functional()
    second = fit_contrast_common_sender_functional(
        source.sample(frac=1.0, random_state=17),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        frozen_interaction_ids=_frozen_ids(source),
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
    assert len(first.contrast_supports) == 1
    assert (
        first.contrast_supports[0].support_id == second.contrast_supports[0].support_id
    )


def test_three_node_context_mapping_is_context_id_order_invariant() -> None:
    functional = _fit_three_context()
    support = functional.contrast_supports[0]

    assert functional.contrast_context_ids == (
        ("a", "z-context"),
        ("b", "a-context"),
        ("c", "m-context"),
    )
    assert functional.contrast_weights == (
        ("a-context", -0.5),
        ("m-context", 1.0),
        ("z-context", -0.5),
    )
    assert support.contrast_weights == functional.contrast_weights
    assert support.subject_effects == pytest.approx((0.50, 0.515, 0.53, 0.545))
    functional.to_dict()


def test_three_node_reordering_preserves_functional_identity() -> None:
    first_contrast = ContrastSpec(
        name="c-v-a-b",
        weights={"a": -0.5, "b": -0.5, "c": 1.0},
        family="three-context-test",
        mode="balanced",
    )
    reordered_contrast = ContrastSpec(
        name="c-v-a-b",
        weights={"c": 1.0, "b": -0.5, "a": -0.5},
        family="three-context-test",
        mode="balanced",
    )
    source = _three_context_availability()
    first = _fit_three_context(first_contrast, source=source)
    reordered = _fit_three_context(
        reordered_contrast,
        source=source.sample(frac=1.0, random_state=2027),
    )

    assert first.contrast.to_dict() == reordered.contrast.to_dict()
    assert first.contrast_context_ids == reordered.contrast_context_ids
    assert first.contrast_weights == reordered.contrast_weights
    assert first.training_availability_digest == (
        reordered.training_availability_digest
    )
    assert first.contrast_supports[0].support_id == (
        reordered.contrast_supports[0].support_id
    )
    assert first.sender_functional_id == reordered.sender_functional_id


def test_three_node_reverse_contrast_preserves_mapping_and_flips_direction() -> None:
    forward = _fit_three_context()
    reverse = _fit_three_context(_three_context_contrast(reverse=True))
    forward_support = forward.contrast_supports[0]
    reverse_support = reverse.contrast_supports[0]

    assert dict(forward.contrast_weights) == {
        "a-context": -0.5,
        "m-context": 1.0,
        "z-context": -0.5,
    }
    assert dict(reverse.contrast_weights) == {
        "a-context": 0.5,
        "m-context": -1.0,
        "z-context": 0.5,
    }
    assert reverse_support.subject_effects == pytest.approx(
        tuple(-value for value in forward_support.subject_effects)
    )
    assert forward_support.mean_effect is not None
    assert reverse_support.mean_effect == pytest.approx(-forward_support.mean_effect)
    assert reverse_support.multiplicity_family_id != (
        forward_support.multiplicity_family_id
    )
    assert reverse.sender_functional_id != forward.sender_functional_id


def test_three_node_equal_weight_mapping_tamper_fails_integrity() -> None:
    functional = _fit_three_context()
    mapping = dict(functional.contrast_context_ids)
    mapping["a"], mapping["b"] = mapping["b"], mapping["a"]
    object.__setattr__(
        functional,
        "contrast_context_ids",
        tuple(sorted(mapping.items())),
    )

    with pytest.raises(ContractError) as error:
        functional.to_dict()
    assert error.value.details.code == ("common_sender_functional_integrity_violation")


def test_two_node_sender_identity_and_effect_parity() -> None:
    functional = _functional()
    support = functional.contrast_supports[0]

    assert functional.sender_functional_id == (
        "contrast_common_sender_functional_595c988d844aab3ddd9df0517c8757e0"
    )
    assert functional.training_availability_digest == (
        "common_sender_training_availability_"
        "8ea5536d219fa571d1870a102c54fb561319aed0dad977cc55ce71c0848ed1a2"
    )
    assert support.support_id == (
        "interaction_ligand_contrast_support_47d4250eadcb03fb6d3fac26303c95c0"
    )
    assert support.multiplicity_family_id == (
        "interaction_ligand_contrast_multiplicity_family_"
        "5b69eb5c2c74eb0dbcb6d217f150c8da"
    )
    assert support.subject_effects == pytest.approx((-0.2, -0.2))


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
        frozen_interaction_ids=_frozen_ids(expanded),
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
            frozen_interaction_ids=_frozen_ids(single),
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


def test_producer_fast_path_is_byte_stable_under_public_revalidation() -> None:
    functional = _functional()
    application = apply_contrast_common_sender_functional(
        functional, _heldout_availability()
    )

    revalidated = CommonSenderApplication(application.table, functional)

    pd.testing.assert_frame_equal(
        application.table, revalidated.table, check_exact=True
    )
    assert application.table["sender_application_id"].tolist() == [
        "common_sender_application_01b6cc127f4c3fd04d2aeaa24e3e28f5",
        "common_sender_application_d0467bc91bc347650b5b0eadcaf18584",
    ]


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
        frozen_interaction_ids=_frozen_ids(changed),
        parameters=first.parameters,
    )
    changed_temperature = fit_contrast_common_sender_functional(
        source,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="filter-universe-1",
        frozen_interaction_ids=_frozen_ids(source),
        parameters=ContrastCommonSenderParameters(
            min_subjects=2,
            prevalence_threshold=0.0,
            softmax_temperature=0.75,
        ),
    )

    assert first.sender_functional_id != changed_prior.sender_functional_id
    assert first.sender_functional_id != changed_temperature.sender_functional_id
    assert first.training_availability_digest != (
        changed_prior.training_availability_digest
    )
    assert first.training_input_digest == "sender-unit-training-input"


def test_ligand_contrast_confidence_is_manifest_bound_and_directional() -> None:
    first_parameters = ContrastCommonSenderParameters(
        min_subjects=2,
        ligand_contrast_confidence_level=0.95,
    )
    changed_parameters = ContrastCommonSenderParameters(
        min_subjects=2,
        ligand_contrast_confidence_level=0.90,
    )
    first = fit_contrast_common_sender_functional(
        _training_availability(),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="confidence-test",
        frozen_interaction_ids=("L_R",),
        parameters=first_parameters,
    )
    changed = fit_contrast_common_sender_functional(
        _training_availability(),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="confidence-test",
        frozen_interaction_ids=("L_R",),
        parameters=changed_parameters,
    )

    assert (
        first_parameters.parameter_manifest_id
        != changed_parameters.parameter_manifest_id
    )
    assert first.sender_functional_id != changed.sender_functional_id
    assert (
        first.contrast_supports[0].support_id != changed.contrast_supports[0].support_id
    )
    for invalid in (0.5, 1.0):
        with pytest.raises(ContractError, match="must lie in"):
            ContrastCommonSenderParameters(ligand_contrast_confidence_level=invalid)
    for invalid_effect in (-0.01, 1.01):
        with pytest.raises(ContractError, match="must lie in"):
            ContrastCommonSenderParameters(
                ligand_contrast_minimum_effect=invalid_effect
            )


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
            frozen_interaction_ids=("L_R",),
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
        frozen_interaction_ids=_frozen_ids(source),
        parameters=ContrastCommonSenderParameters(min_subjects=2),
    )
    application = apply_contrast_common_sender_functional(
        functional, _heldout_availability()
    )
    result: pd.DataFrame = application.table.set_index("sender")

    prior_value: Any = result.loc["B", "training_prevalence_prior"]
    assert pd.isna(prior_value)
    assert result["assignment_weight"].isna().all()
    assert set(result["status"]) == {CommonSenderApplicationStatus.NOT_ESTIMABLE.value}
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
            frozen_interaction_ids=_frozen_ids(training),
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
    omitted_candidate = application.table.loc[application.table["sender"].eq("A")].copy(
        deep=True
    )
    omitted_candidate["assignment_weight"] = 1.0
    omitted_candidate["normalized_entropy"] = 0.0

    with pytest.raises(ContractError, match="outside the bound contrast"):
        CommonSenderApplication(wrong_context, application.functional)
    with pytest.raises(ContractError, match="frozen sender universe"):
        CommonSenderApplication(omitted_candidate, application.functional)


def test_active_like_interaction_has_positive_one_sided_support() -> None:
    effects = (0.40, 0.45, 0.50, 0.42)
    functional = _fit_paired_interaction(effects)
    support = functional.contrast_supports[0]

    assert functional.schema_version == "3.0.0"
    assert functional.parameters.schema_version == "3.0.0"
    assert support.aggregation_policy == (
        "subject_equal_paired_one_sided_t_holm_fwer_v3"
    )
    assert support.complete_subject_ids == ("s1", "s2", "s3", "s4")
    assert support.subject_effects == pytest.approx(effects)
    assert support.n_complete == 4
    assert support.degrees_of_freedom == 3
    assert support.mean_effect == pytest.approx(np.mean(effects))
    assert support.critical_value == pytest.approx(student_t.ppf(0.95, 3))
    assert support.lower_confidence_bound is not None
    assert support.lower_confidence_bound > 0
    assert support.ligand_contrast_minimum_effect == 0.0
    assert support.raw_one_sided_p_value is not None
    assert support.raw_one_sided_p_value < 0.05
    assert support.holm_adjusted_p_value == pytest.approx(support.raw_one_sided_p_value)
    assert support.holm_rank == 1
    assert support.multiplicity_family_size == 1
    assert support.multiplicity_method == "holm_step_down"
    assert support.status is SenderContrastSupportStatus.SUPPORTED
    assert support.reason_code is None

    gate = interaction_ligand_contrast_gate(functional, "R", "L_R")
    assert gate.gate == 1.0
    assert gate.status is SenderContrastSupportStatus.SUPPORTED
    assert gate.support_ids == (support.support_id,)
    assert gate.to_dict()["gate_id"] == gate.gate_id


def test_target_like_noisy_interaction_is_observed_but_unsupported() -> None:
    effects = (-0.04, 0.04, -0.02, 0.02, 0.00, 0.01)
    functional = _fit_paired_interaction(effects)
    support = functional.contrast_supports[0]

    assert support.n_complete == len(effects)
    assert support.mean_effect == pytest.approx(np.mean(effects))
    assert support.sample_standard_error is not None
    assert support.sample_standard_error > 0
    assert support.lower_confidence_bound is not None
    assert support.lower_confidence_bound < 0
    assert support.status is SenderContrastSupportStatus.UNSUPPORTED
    assert support.reason_code == (
        "ligand_contrast_holm_adjusted_p_not_below_familywise_alpha"
    )

    gate = interaction_ligand_contrast_gate(functional, "R", "L_R")
    assert gate.gate == 0.0
    assert gate.status is SenderContrastSupportStatus.UNSUPPORTED
    assert gate.reason_code == support.reason_code
    assert gate.support_ids == (support.support_id,)


def test_interaction_support_uses_one_max_across_candidate_senders() -> None:
    source = _paired_interaction_availability((0.4, 0.4, 0.4))
    duplicated = source.loc[source["sender"].eq("B")].copy(deep=True)
    duplicated["sender"] = "C"
    duplicated["sample_id"] = duplicated["sample_id"].str.replace(
        "-B", "-C", regex=False
    )
    functional = fit_contrast_common_sender_functional(
        pd.concat([source, duplicated], ignore_index=True),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="candidate-count-test",
        frozen_interaction_ids=("L_R",),
        parameters=ContrastCommonSenderParameters(min_subjects=2),
    )

    assert len(functional.candidate_priors) == 3
    assert len(functional.contrast_supports) == 1
    assert functional.contrast_supports[0].subject_effects == pytest.approx(
        (0.4, 0.4, 0.4)
    )


def test_zero_variance_lower_bound_equals_mean() -> None:
    functional = _fit_paired_interaction((0.25, 0.25, 0.25))
    support = functional.contrast_supports[0]

    assert support.sample_standard_error == 0.0
    assert support.mean_effect == pytest.approx(0.25)
    assert support.lower_confidence_bound == support.mean_effect
    assert support.raw_one_sided_p_value == 0.0
    assert support.holm_adjusted_p_value == 0.0
    assert support.status is SenderContrastSupportStatus.SUPPORTED


def test_missing_context_and_low_complete_support_are_not_estimable() -> None:
    functional = _fit_paired_interaction(
        (0.4, 0.4, 0.4),
        min_subjects=3,
        missing_treated_subjects=("s3",),
    )
    support = functional.contrast_supports[0]

    assert support.complete_subject_ids == ("s1", "s2")
    assert support.subject_effects == pytest.approx((0.4, 0.4))
    assert support.n_complete == 2
    assert support.minimum_complete_subjects == 3
    assert support.mean_effect is None
    assert support.sample_standard_error is None
    assert support.lower_confidence_bound is None
    assert support.raw_one_sided_p_value is None
    assert support.holm_adjusted_p_value == 1.0
    assert support.holm_rank == 1
    assert support.multiplicity_family_size == 1
    assert support.status is SenderContrastSupportStatus.NOT_ESTIMABLE
    assert support.reason_code == ("insufficient_complete_subject_ligand_contrasts")

    gate = interaction_ligand_contrast_gate(functional, "R", "L_R")
    assert gate.gate is None
    assert gate.status is SenderContrastSupportStatus.NOT_ESTIMABLE
    assert gate.support_ids == (support.support_id,)


def test_seed3_like_raw_positive_is_removed_by_five_interaction_holm_family() -> None:
    functional = _fit_multi_interactions(
        {
            "EGF_EGFR": (0.12096, 0.03778, 0.07275, 0.07245),
            "NULL_A": (-0.04, 0.04, -0.02, 0.02),
            "NULL_B": (-0.10, -0.05, 0.02, 0.01),
            "NULL_C": (0.0, 0.0, 0.0, 0.0),
            "NULL_D": (0.02, -0.01, 0.01, -0.02),
        }
    )
    supports = {
        support.interaction_id: support for support in functional.contrast_supports
    }
    egf = supports["EGF_EGFR"]

    assert egf.raw_one_sided_p_value == pytest.approx(0.01058371, rel=1e-6)
    assert egf.lower_confidence_bound is not None
    assert egf.lower_confidence_bound > 0
    assert egf.multiplicity_family_size == 5
    assert egf.holm_rank == 1
    assert egf.holm_adjusted_p_value == pytest.approx(5 * egf.raw_one_sided_p_value)
    assert egf.holm_adjusted_p_value > 0.05
    assert egf.status is SenderContrastSupportStatus.UNSUPPORTED
    assert len({support.multiplicity_family_id for support in supports.values()}) == 1


def test_strong_active_like_interactions_survive_holm_with_null_candidates() -> None:
    functional = _fit_multi_interactions(
        {
            "ACTIVE_A": (0.40, 0.45, 0.50, 0.42),
            "ACTIVE_B": (0.35, 0.40, 0.38, 0.44),
            "ACTIVE_C": (0.30, 0.33, 0.36, 0.39),
            "NULL_A": (-0.04, 0.04, -0.02, 0.02),
            "NULL_B": (0.02, -0.01, 0.01, -0.02),
        }
    )
    supports = {
        support.interaction_id: support for support in functional.contrast_supports
    }

    for interaction_id in ("ACTIVE_A", "ACTIVE_B", "ACTIVE_C"):
        support = supports[interaction_id]
        assert support.multiplicity_family_size == 5
        assert support.holm_adjusted_p_value < 0.05
        assert support.status is SenderContrastSupportStatus.SUPPORTED
    for interaction_id in ("NULL_A", "NULL_B"):
        assert (
            supports[interaction_id].status is SenderContrastSupportStatus.UNSUPPORTED
        )


def test_not_estimable_interaction_does_not_shrink_holm_family() -> None:
    functional = _fit_multi_interactions(
        {
            "EGF_EGFR": (0.12096, 0.03778, 0.07275, 0.07245),
            "NULL_A": (-0.04, 0.04, -0.02, 0.02),
            "NULL_B": (-0.10, -0.05, 0.02, 0.01),
            "NULL_C": (0.02, -0.01, 0.01, -0.02),
            "NOT_ESTIMABLE": (0.40, 0.40, 0.40, 0.40),
        },
        min_subjects=3,
        missing_treated_subjects={"NOT_ESTIMABLE": ("s3", "s4")},
    )
    supports = {
        support.interaction_id: support for support in functional.contrast_supports
    }
    egf = supports["EGF_EGFR"]
    not_estimable = supports["NOT_ESTIMABLE"]

    assert egf.multiplicity_family_size == 5
    assert egf.holm_adjusted_p_value == pytest.approx(
        5 * egf.raw_one_sided_p_value  # type: ignore[operator]
    )
    assert egf.status is SenderContrastSupportStatus.UNSUPPORTED
    assert not_estimable.raw_one_sided_p_value is None
    assert not_estimable.holm_adjusted_p_value == 1.0
    assert not_estimable.multiplicity_family_size == 5
    assert not_estimable.status is SenderContrastSupportStatus.NOT_ESTIMABLE


def test_completely_missing_frozen_interactions_remain_in_holm_family() -> None:
    interaction_effects = {
        "EGF_EGFR": (0.12096, 0.03778, 0.07275, 0.07245),
        "NULL_A": (-0.04, 0.04, -0.02, 0.02),
        "NULL_B": (-0.10, -0.05, 0.02, 0.01),
        "NULL_C": (0.02, -0.01, 0.01, -0.02),
        "NO_ROWS": (0.40, 0.40, 0.40, 0.40),
    }
    full = _multi_interaction_availability(interaction_effects)
    frozen_ids = tuple(sorted(interaction_effects))
    manifest = freeze_common_sender_candidate_manifest(
        full,
        frozen_interaction_ids=frozen_ids,
    )
    partial = full.loc[full["interaction_id"].eq("EGF_EGFR")].copy()

    functional = fit_contrast_common_sender_functional(
        partial,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="zero-row-family-test",
        frozen_interaction_ids=frozen_ids,
        frozen_candidate_sender_manifest=manifest,
        parameters=ContrastCommonSenderParameters(min_subjects=3),
    )
    supports = {
        support.interaction_id: support for support in functional.contrast_supports
    }

    assert set(supports) == set(frozen_ids)
    assert supports["EGF_EGFR"].multiplicity_family_size == 5
    assert supports["EGF_EGFR"].holm_adjusted_p_value == pytest.approx(
        5 * supports["EGF_EGFR"].raw_one_sided_p_value  # type: ignore[operator]
    )
    assert supports["EGF_EGFR"].status is SenderContrastSupportStatus.UNSUPPORTED
    for interaction_id in set(frozen_ids).difference({"EGF_EGFR"}):
        support = supports[interaction_id]
        assert support.raw_one_sided_p_value is None
        assert support.holm_adjusted_p_value == 1.0
        assert support.status is SenderContrastSupportStatus.NOT_ESTIMABLE
    missing_priors = [
        prior
        for prior in functional.candidate_priors
        if prior.interaction_id != "EGF_EGFR"
    ]
    assert missing_priors
    assert all(prior.n_subjects == 0 for prior in missing_priors)
    assert all(
        prior.status is SenderPrevalenceStatus.MISSING_EVIDENCE
        for prior in missing_priors
    )


def test_completely_missing_receiver_remains_frozen_and_not_estimable() -> None:
    base = _multi_interaction_availability(
        {
            "ACTIVE": (0.40, 0.45, 0.50, 0.42),
            "NULL": (-0.04, 0.04, -0.02, 0.02),
        }
    )
    second_receiver = base.copy(deep=True)
    second_receiver["receiver"] = "R2"
    full = pd.concat([base, second_receiver], ignore_index=True)
    frozen_ids = ("ACTIVE", "NULL")
    manifest = freeze_common_sender_candidate_manifest(
        full,
        frozen_interaction_ids=frozen_ids,
    )

    functional = fit_contrast_common_sender_functional(
        base,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="missing-receiver-family-test",
        frozen_interaction_ids=frozen_ids,
        frozen_candidate_sender_manifest=manifest,
        parameters=ContrastCommonSenderParameters(min_subjects=2),
    )
    by_receiver = {
        receiver: tuple(
            support
            for support in functional.contrast_supports
            if support.receiver == receiver
        )
        for receiver in ("R", "R2")
    }

    assert len(by_receiver["R"]) == len(by_receiver["R2"]) == 2
    assert all(support.multiplicity_family_size == 2 for support in by_receiver["R2"])
    assert all(
        support.status is SenderContrastSupportStatus.NOT_ESTIMABLE
        for support in by_receiver["R2"]
    )
    assert {support.multiplicity_family_id for support in by_receiver["R"]}.isdisjoint(
        {support.multiplicity_family_id for support in by_receiver["R2"]}
    )


def test_training_rows_outside_frozen_candidate_manifest_are_rejected() -> None:
    source = _paired_interaction_availability((0.4, 0.4, 0.4))
    frozen_ids = ("L_R",)
    manifest = freeze_common_sender_candidate_manifest(
        source,
        frozen_interaction_ids=frozen_ids,
    )
    poison = source.iloc[[0]].copy(deep=True)
    poison["sample_id"] = poison["sample_id"].astype(str) + "-poison"
    poison["sender"] = "UNFROZEN"

    with pytest.raises(ContractError, match="outside the frozen sender manifest"):
        fit_contrast_common_sender_functional(
            pd.concat([source, poison], ignore_index=True),
            contrast=_contrast(),
            context_keys=("context_id",),
            filter_universe_id="candidate-manifest-poison-test",
            frozen_interaction_ids=frozen_ids,
            frozen_candidate_sender_manifest=manifest,
            parameters=ContrastCommonSenderParameters(min_subjects=2),
        )


def test_multi_interaction_holm_family_is_row_order_and_tie_break_stable() -> None:
    source = _multi_interaction_availability(
        {
            "TIE_B": (0.40, 0.40, 0.40, 0.40),
            "TIE_A": (0.40, 0.40, 0.40, 0.40),
            "NULL": (-0.02, 0.02, -0.01, 0.01),
        }
    )
    parameters = ContrastCommonSenderParameters(min_subjects=2)
    manifest = freeze_common_sender_candidate_manifest(
        source,
        frozen_interaction_ids=_frozen_ids(source),
    )
    first = fit_contrast_common_sender_functional(
        source,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="multi-order-test",
        frozen_interaction_ids=_frozen_ids(source),
        frozen_candidate_sender_manifest=tuple(reversed(manifest)),
        parameters=parameters,
    )
    shuffled = fit_contrast_common_sender_functional(
        source.sample(frac=1.0, random_state=1414),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="multi-order-test",
        frozen_interaction_ids=_frozen_ids(source),
        frozen_candidate_sender_manifest=json.loads(json.dumps(manifest)),
        parameters=parameters,
    )
    first_supports = {
        support.interaction_id: support for support in first.contrast_supports
    }
    shuffled_supports = {
        support.interaction_id: support for support in shuffled.contrast_supports
    }

    assert first_supports["TIE_A"].holm_rank == 1
    assert first_supports["TIE_B"].holm_rank == 2
    assert {key: value.support_id for key, value in first_supports.items()} == {
        key: value.support_id for key, value in shuffled_supports.items()
    }
    assert first.sender_functional_id == shuffled.sender_functional_id


def test_minimum_effect_is_configurable_but_defaults_to_zero() -> None:
    effects = (0.12096, 0.03778, 0.07275, 0.07245)
    default = _fit_paired_interaction(effects)
    practical = _fit_paired_interaction(
        effects,
        ligand_contrast_minimum_effect=0.05,
    )
    default_support = default.contrast_supports[0]
    practical_support = practical.contrast_supports[0]

    assert default.parameters.ligand_contrast_minimum_effect == 0.0
    assert default_support.status is SenderContrastSupportStatus.SUPPORTED
    assert practical_support.ligand_contrast_minimum_effect == 0.05
    assert practical_support.raw_one_sided_p_value == pytest.approx(
        0.1129,
        rel=1e-3,
    )
    assert practical_support.status is SenderContrastSupportStatus.UNSUPPORTED
    assert default.parameters.parameter_manifest_id != (
        practical.parameters.parameter_manifest_id
    )
    assert default.sender_functional_id != practical.sender_functional_id


def test_holm_family_id_binds_minimum_subjects_and_candidate_senders() -> None:
    effects = (0.40, 0.45, 0.50, 0.42)
    minimum_two = _fit_paired_interaction(effects, min_subjects=2)
    minimum_four = _fit_paired_interaction(effects, min_subjects=4)
    assert minimum_two.contrast_supports[0].multiplicity_family_id != (
        minimum_four.contrast_supports[0].multiplicity_family_id
    )

    source = _paired_interaction_availability(effects)
    extra_sender = source.loc[source["sender"].eq("B")].copy(deep=True)
    extra_sender["sender"] = "C"
    extra_sender["sample_id"] = extra_sender["sample_id"].str.replace(
        "-B", "-C", regex=False
    )
    expanded = fit_contrast_common_sender_functional(
        pd.concat([source, extra_sender], ignore_index=True),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="interaction-support-test",
        frozen_interaction_ids=("L_R",),
        parameters=minimum_two.parameters,
    )
    assert minimum_two.contrast_supports[0].multiplicity_family_id != (
        expanded.contrast_supports[0].multiplicity_family_id
    )


def test_functional_rejects_coordinated_support_replacement_across_inputs() -> None:
    weak = _fit_paired_interaction((0.0, 0.0, 0.0, 0.0))
    strong = _fit_paired_interaction((0.4, 0.4, 0.4, 0.4))
    assert weak.candidate_priors == strong.candidate_priors
    assert weak.training_availability_digest != strong.training_availability_digest

    object.__setattr__(weak, "contrast_supports", strong.contrast_supports)
    object.__setattr__(weak, "sender_functional_id", strong.sender_functional_id)

    with pytest.raises(ContractError) as error:
        weak.to_dict()
    assert error.value.details.code == ("common_sender_functional_integrity_violation")


@pytest.mark.parametrize(
    ("effect", "minimum_effect"),
    ((0.05, 0.05), (0.04, 0.05)),
)
def test_zero_variance_effect_at_or_below_minimum_is_unsupported(
    effect: float,
    minimum_effect: float,
) -> None:
    functional = _fit_paired_interaction(
        (effect, effect, effect, effect),
        ligand_contrast_minimum_effect=minimum_effect,
    )
    support = functional.contrast_supports[0]

    assert support.sample_standard_error == 0.0
    assert support.raw_one_sided_p_value == 1.0
    assert support.holm_adjusted_p_value == 1.0
    assert support.status is SenderContrastSupportStatus.UNSUPPORTED


def test_holm_support_requires_adjusted_p_strictly_below_alpha() -> None:
    support = _fit_paired_interaction((0.40, 0.45, 0.50, 0.42)).contrast_supports[0]
    boundary = _rebuilt_support(
        support,
        holm_adjusted_p_value=0.05,
        status=SenderContrastSupportStatus.UNSUPPORTED,
        reason_code=("ligand_contrast_holm_adjusted_p_not_below_familywise_alpha"),
    )

    assert boundary.holm_adjusted_p_value == 0.05
    assert boundary.status is SenderContrastSupportStatus.UNSUPPORTED


@pytest.mark.parametrize("forgery", ["family", "rank", "adjusted_p"])
def test_functional_jointly_rejects_forged_holm_family(forgery: str) -> None:
    functional = _fit_multi_interactions(
        {
            "ACTIVE_A": (0.40, 0.45, 0.50, 0.42),
            "ACTIVE_B": (0.35, 0.40, 0.38, 0.44),
            "NULL": (-0.04, 0.04, -0.02, 0.02),
        }
    )
    original = next(
        support for support in functional.contrast_supports if support.holm_rank == 1
    )
    if forgery == "family":
        forged = _rebuilt_support(
            original,
            multiplicity_family_id="forged-family-id",
        )
    elif forgery == "rank":
        forged = _rebuilt_support(original, holm_rank=2)
    else:
        forged = _rebuilt_support(
            original,
            holm_adjusted_p_value=0.04,
        )
    supports = tuple(
        forged if support.interaction_id == original.interaction_id else support
        for support in functional.contrast_supports
    )

    object.__setattr__(functional, "contrast_supports", supports)
    with pytest.raises(ContractError) as error:
        functional.to_dict()
    assert error.value.details.code == ("common_sender_functional_integrity_violation")


def test_technical_rows_are_meaned_before_interaction_sender_maximum() -> None:
    rows: list[dict[str, object]] = []
    values = {
        ("s1", "control", "A"): (0.1, 0.3),
        ("s1", "control", "B"): (0.4, 0.2),
        ("s1", "treated", "A"): (0.8, 0.6),
        ("s1", "treated", "B"): (0.5, 0.5),
        ("s2", "control", "A"): (0.2, 0.4),
        ("s2", "control", "B"): (0.1, 0.1),
        ("s2", "treated", "A"): (0.5, 0.5),
        ("s2", "treated", "B"): (0.7, 0.7),
    }
    for (subject, context, sender), technical_values in values.items():
        for technical_index, ligand in enumerate(technical_values):
            rows.append(
                {
                    "sample_id": (f"{subject}-{context}-{sender}-{technical_index}"),
                    "subject_id": subject,
                    "context_id": context,
                    "sender": sender,
                    "receiver": "R",
                    "interaction_id": "L_R",
                    "ligand_availability": ligand,
                }
            )
    source = pd.DataFrame(rows)
    first = fit_contrast_common_sender_functional(
        source,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="technical-row-test",
        frozen_interaction_ids=("L_R",),
        parameters=ContrastCommonSenderParameters(min_subjects=2),
    )
    shuffled = fit_contrast_common_sender_functional(
        source.sample(frac=1.0, random_state=2026),
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="technical-row-test",
        frozen_interaction_ids=("L_R",),
        parameters=first.parameters,
    )

    # s1: max(mean(A), mean(B)) is 0.3 -> 0.7; s2 is 0.3 -> 0.7.
    support = first.contrast_supports[0]
    assert support.subject_effects == pytest.approx((0.4, 0.4))
    assert support.support_id == shuffled.contrast_supports[0].support_id
    assert first.sender_functional_id == shuffled.sender_functional_id
    assert len(first.candidate_priors) == 2
    assert len(first.contrast_supports) == 1


def test_absent_interaction_gate_is_not_estimable_without_support_id() -> None:
    gate = interaction_ligand_contrast_gate(
        _fit_paired_interaction((0.4, 0.4)), "R", "not-frozen"
    )

    assert gate.gate is None
    assert gate.status is SenderContrastSupportStatus.NOT_ESTIMABLE
    assert gate.reason_code == "interaction_ligand_contrast_support_absent"
    assert gate.support_ids == ()


def test_bulk_interaction_gates_match_ordered_single_gate_results() -> None:
    functional = _fit_paired_interaction((0.4, 0.45, 0.5))
    queries = (("R", "not-frozen"), ("R", "L_R"), ("R", "L_R"))

    bulk = interaction_ligand_contrast_gates(functional, queries)
    singles = tuple(
        interaction_ligand_contrast_gate(functional, receiver, interaction_id)
        for receiver, interaction_id in queries
    )

    assert tuple(gate.to_dict() for gate in bulk) == tuple(
        gate.to_dict() for gate in singles
    )


def test_support_gate_and_functional_reject_forced_mutation() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        InteractionLigandContrastSupport()

    support_functional = _fit_paired_interaction((0.4, 0.45, 0.5))
    support = support_functional.contrast_supports[0]
    object.__setattr__(support, "lower_confidence_bound", -1.0)
    with pytest.raises(ContractError) as support_error:
        support.to_dict()
    assert support_error.value.details.code == (
        "sender_contrast_support_integrity_violation"
    )
    with pytest.raises(ContractError) as parent_error:
        support_functional.to_dict()
    assert parent_error.value.details.code == (
        "common_sender_functional_integrity_violation"
    )

    gate = interaction_ligand_contrast_gate(
        _fit_paired_interaction((0.4, 0.45, 0.5)), "R", "L_R"
    )
    object.__setattr__(gate, "gate", 0.0)
    with pytest.raises(ContractError) as gate_error:
        gate.to_dict()
    assert gate_error.value.details.code == (
        "interaction_ligand_contrast_gate_integrity_violation"
    )

    functional = _fit_paired_interaction((0.4, 0.45, 0.5))
    object.__setattr__(functional, "contrast_supports", ())
    with pytest.raises(ContractError) as functional_error:
        functional.to_dict()
    assert functional_error.value.details.code == (
        "common_sender_functional_integrity_violation"
    )


def _independent_availability(
    *,
    include_treated: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {
        "control": {"c1": 0.20, "c2": 0.25, "c3": 0.30},
        "treated": {"t1": 0.80, "t2": 0.75, "t3": 0.85},
    }
    if not include_treated:
        groups.pop("treated")
    for context, subjects in groups.items():
        for subject, value in subjects.items():
            for sender, ligand in (("A", value), ("B", value - 0.1)):
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


def _fit_independent(
    source: pd.DataFrame,
) -> ContrastCommonSenderFunctional:
    return fit_contrast_common_sender_functional(
        source,
        contrast=_contrast(),
        context_keys=("context_id",),
        filter_universe_id="independent-unit-test",
        frozen_interaction_ids=("L_R",),
        parameters=ContrastCommonSenderParameters(
            min_subjects=2,
            contrast_unit="independent_subject",
        ),
    )


def test_independent_subject_contrast_is_estimable_without_pairing() -> None:
    functional = _fit_independent(_independent_availability())
    support = functional.contrast_supports[0]

    assert support.aggregation_policy == (
        "subject_equal_independent_welch_one_sided_t_holm_fwer_v1"
    )
    assert support.n_complete == 6
    assert len(support.unit_context_ids) == 6
    assert support.status is SenderContrastSupportStatus.SUPPORTED
    assert support.mean_effect == pytest.approx(0.55, abs=1e-12)
    assert interaction_ligand_contrast_gate(functional, "R", "L_R").gate == 1.0
    functional._require_intact()


def test_independent_subject_contrast_marks_missing_group_not_estimable() -> None:
    source = _independent_availability()
    source = source.loc[~source["subject_id"].isin(("t2", "t3"))]
    functional = _fit_independent(source)
    support = functional.contrast_supports[0]

    assert support.status is SenderContrastSupportStatus.NOT_ESTIMABLE
    assert support.reason_code == "insufficient_independent_subject_ligand_contrasts"
    assert support.mean_effect is None
    assert interaction_ligand_contrast_gate(functional, "R", "L_R").gate is None
    functional._require_intact()


def test_independent_subject_contrast_rejects_global_overlap_before_na_filter() -> None:
    source = _independent_availability()
    # Keep the overlap only in rows that would not contribute to one of the
    # interaction summaries.  The design contract must still reject it.
    source.loc[source["subject_id"].eq("t1"), "subject_id"] = "c1"
    source.loc[
        source["subject_id"].eq("c1") & source["context_id"].eq("treated"),
        "ligand_availability",
    ] = np.nan

    with pytest.raises(ContractError, match="disjoint subject IDs") as error:
        _fit_independent(source)
    assert error.value.details.code == "invalid_common_sender_input"


def test_independent_subject_contrast_uses_exact_welch_statistics() -> None:
    rows: list[dict[str, object]] = []
    values = {
        "control": {"c1": 0.10, "c2": 0.20},
        "treated": {"t1": 0.70, "t2": 0.80, "t3": 1.00},
    }
    for context, subjects in values.items():
        for subject, value in subjects.items():
            rows.extend(
                {
                    "sample_id": f"{subject}-{context}-{sender}",
                    "subject_id": subject,
                    "context_id": context,
                    "sender": sender,
                    "receiver": "R",
                    "interaction_id": "L_R",
                    "ligand_availability": ligand,
                }
                for sender, ligand in (("A", value), ("B", value - 0.1))
            )
    functional = _fit_independent(pd.DataFrame(rows))
    support = functional.contrast_supports[0]
    control = np.array([0.10, 0.20])
    treated = np.array([0.70, 0.80, 1.00])
    expected_mean = float(treated.mean() - control.mean())
    expected_variance = float(treated.var(ddof=1) / len(treated)) + float(
        control.var(ddof=1) / len(control)
    )
    expected_se = math.sqrt(expected_variance)
    expected_df = expected_variance**2 / (
        (treated.var(ddof=1) / len(treated)) ** 2 / (len(treated) - 1)
        + (control.var(ddof=1) / len(control)) ** 2 / (len(control) - 1)
    )
    expected_p = float(student_t.sf(expected_mean / expected_se, expected_df))

    assert support.mean_effect == pytest.approx(expected_mean)
    assert support.sample_standard_error == pytest.approx(expected_se)
    assert support.degrees_of_freedom == pytest.approx(expected_df)
    assert support.degrees_of_freedom != int(support.degrees_of_freedom)
    assert support.raw_one_sided_p_value == pytest.approx(expected_p)
    shuffled = _fit_independent(pd.DataFrame(rows).sample(frac=1, random_state=19))
    assert shuffled.contrast_supports[0].support_id == support.support_id


def test_independent_subject_parameter_rejects_unknown_sampling_unit() -> None:
    with pytest.raises(ContractError, match="contrast_unit"):
        ContrastCommonSenderParameters(contrast_unit="mixed_subject")
