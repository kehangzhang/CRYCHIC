from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.scoring import (
    CORE_COMPONENTS,
    CommunicationScores,
    CommunicationScoreStatus,
    ScoringFunctional,
    ScoringFunctionalStatus,
    score_communication,
    validate_common_functional,
    weighted_geometric_strength,
)


def _functional(
    *,
    status: ScoringFunctionalStatus = ScoringFunctionalStatus.EXPLORATORY_IN_SAMPLE,
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> ScoringFunctional:
    return ScoringFunctional(
        contrast_name="treated-v-control",
        contrast_contexts=("treated", "control"),
        training_subject_ids=("train-2", "train-1"),
        interaction_ids=("i2", "i1"),
        target_ids=("G2", "G1"),
        status=status,
        fold_id="fold-1" if status is ScoringFunctionalStatus.OUT_OF_FOLD else None,
        component_weights=weights or dict.fromkeys(CORE_COMPONENTS, 1.0),
        component_scales=scales or dict.fromkeys(CORE_COMPONENTS, 1.0),
    )


def _availability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-c", "sample-t"],
            "subject_id": ["test-c", "test-t"],
            "context": ["control", "treated"],
            "sender": ["S", "S"],
            "receiver": ["R", "R"],
            "interaction_id": ["i1", "i1"],
            "state_availability": [0.8, 0.8],
            "ecosystem_availability": [0.2, 0.8],
            "sender_component": [0.5, 0.5],
            "abundance_component": [0.25, 1.0],
        }
    )


def _downstream() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-c", "sample-t"],
            "subject_id": ["test-c", "test-t"],
            "context": ["control", "treated"],
            "receiver": ["R", "R"],
            "interaction_id": ["i1", "i1"],
            "downstream_activity": [0.5, 0.5],
            "prior_quality": [1.0, 1.0],
        }
    )


def _tagged(table: pd.DataFrame, functional: ScoringFunctional) -> pd.DataFrame:
    return table.assign(scoring_function_id=functional.scoring_function_id)


def test_functional_id_is_stable_under_universe_reordering() -> None:
    first = _functional()
    second = ScoringFunctional(
        contrast_name="treated-v-control",
        contrast_contexts=("control", "treated"),
        training_subject_ids=("train-1", "train-2"),
        interaction_ids=("i1", "i2"),
        target_ids=("G1", "G2"),
    )

    assert first.scoring_function_id == second.scoring_function_id
    assert first.interaction_universe_id == second.interaction_universe_id
    assert first.target_universe_id == second.target_universe_id
    assert first.reason_code == "exploratory_not_cross_fitted"
    assert first.frozen
    assert first.to_dict()["output_scale"] == "unit_interval"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"contrast_name": ""}, "contrast_name"),
        ({"contrast_contexts": ("control",)}, "at least two"),
        ({"contrast_contexts": ("control", "control")}, "unique"),
        ({"training_subject_ids": ()}, "must not be empty"),
        ({"interaction_ids": ("i1", "i1")}, "unique"),
        ({"target_ids": ("",)}, "non-empty strings"),
        ({"frozen": False}, "must be frozen"),
        ({"output_scale": "raw_product"}, "unit_interval"),
        (
            {
                "status": ScoringFunctionalStatus.OUT_OF_FOLD,
                "fold_id": None,
            },
            "requires a non-empty fold_id",
        ),
        ({"fold_id": "fold-on-in-sample"}, "must not declare"),
        ({"component_weights": {"availability": 1.0}}, "exactly"),
        ({"component_weights": dict.fromkeys(CORE_COMPONENTS, 0.0)}, "positive"),
        ({"component_scales": dict.fromkeys(CORE_COMPONENTS, 0.0)}, "positive"),
    ],
)
def test_invalid_functional_contracts_are_rejected(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "contrast_name": "treated-v-control",
        "contrast_contexts": ("control", "treated"),
        "training_subject_ids": ("train-1", "train-2"),
        "interaction_ids": ("i1",),
        "target_ids": ("G1",),
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        ScoringFunctional(**values)  # type: ignore[arg-type]


def test_weighted_geometric_strength_matches_hand_calculation_and_scale() -> None:
    components = {
        "availability": 0.25,
        "downstream": 0.5,
        "sender": 1.0,
        "prior_quality": 1.0,
    }
    weights = dict.fromkeys(CORE_COMPONENTS, 1.0)

    observed = weighted_geometric_strength(components, weights)

    assert observed == pytest.approx((0.25 * 0.5) ** 0.25)
    assert weighted_geometric_strength(
        dict.fromkeys(CORE_COMPONENTS, 0.5),
        weights,
        dict.fromkeys(CORE_COMPONENTS, 0.5),
    ) == pytest.approx(1.0)
    assert (
        weighted_geometric_strength({**components, "availability": 0.0}, weights) == 0.0
    )


def test_geometric_contract_rejects_mismatched_or_invalid_parameters() -> None:
    components = dict.fromkeys(CORE_COMPONENTS, 0.5)
    weights = dict.fromkeys(CORE_COMPONENTS, 1.0)
    with pytest.raises(ValueError, match="identical keys"):
        weighted_geometric_strength(components, {"availability": 1.0})
    with pytest.raises(ValueError, match="same keys"):
        weighted_geometric_strength(components, weights, {"availability": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        weighted_geometric_strength(components, {**weights, "availability": -1.0})
    with pytest.raises(ValueError, match="finite and positive"):
        weighted_geometric_strength(
            components,
            weights,
            {**dict.fromkeys(CORE_COMPONENTS, 1.0), "availability": 0.0},
        )


@pytest.mark.parametrize("component", CORE_COMPONENTS)
def test_strength_is_monotone_in_each_positive_weight_component(component: str) -> None:
    weights = dict.fromkeys(CORE_COMPONENTS, 1.0)
    low_components = dict.fromkeys(CORE_COMPONENTS, 0.25)
    high_components = dict(low_components)
    high_components[component] = 0.75

    low = weighted_geometric_strength(low_components, weights)
    high = weighted_geometric_strength(high_components, weights)

    assert low is not None and high is not None
    assert high > low


def test_state_and_ecosystem_are_separate_and_components_are_preserved() -> None:
    functional = _functional()
    availability = _availability()
    downstream = _downstream()
    before_availability = availability.copy(deep=True)
    before_downstream = downstream.copy(deep=True)

    result = score_communication(availability, downstream, functional)

    assert len(result.table) == 4
    state = result.for_mode("state").set_index("context")
    ecosystem = result.for_mode("ecosystem").set_index("context")
    expected_state = (0.8 * 0.5 * 0.5 * 1.0) ** 0.25
    assert state.loc["control", "comm_strength"] == pytest.approx(expected_state)
    assert state.loc["treated", "comm_strength"] == pytest.approx(expected_state)
    assert (
        ecosystem.loc["treated", "comm_strength"]
        > ecosystem.loc["control", "comm_strength"]
    )
    assert state.loc["control", "availability"] == pytest.approx(0.8)
    assert state.loc["control", "downstream_activity"] == pytest.approx(0.5)
    assert state.loc["control", "sender_component"] == pytest.approx(0.5)
    assert state.loc["control", "prior_quality"] == pytest.approx(1.0)
    assert state.loc["control", "abundance_component"] == pytest.approx(0.25)
    assert set(result.table["functional_reason_code"]) == {
        "exploratory_not_cross_fitted"
    }
    assert result.score_semantics == "strength_not_probability"
    pd.testing.assert_frame_equal(availability, before_availability)
    pd.testing.assert_frame_equal(downstream, before_downstream)


def test_missing_core_evidence_propagates_na_but_zero_is_valid() -> None:
    availability = _availability()
    availability.loc[0, "state_availability"] = np.nan
    availability.loc[1, "state_availability"] = 0.0
    result = score_communication(availability, _downstream(), _functional())
    state = result.for_mode("state").set_index("context")

    assert np.isnan(state.loc["control", "comm_strength"])
    assert state.loc["control", "status"] == (
        CommunicationScoreStatus.MISSING_CORE_EVIDENCE.value
    )
    assert state.loc["control", "reason_code"] == ("missing_core_evidence:availability")
    assert state.loc["treated", "comm_strength"] == 0.0
    assert state.loc["treated", "status"] == CommunicationScoreStatus.OK.value


def test_missing_downstream_join_does_not_become_zero() -> None:
    result = score_communication(
        _availability(), _downstream().iloc[[0]].copy(), _functional()
    )
    treated = result.table[result.table["context"] == "treated"]

    assert treated["comm_strength"].isna().all()
    assert set(treated["reason_code"]) == {
        "missing_core_evidence:downstream,prior_quality"
    }


def test_zero_weight_ablation_does_not_require_missing_component() -> None:
    weights = dict.fromkeys(CORE_COMPONENTS, 1.0)
    weights["sender"] = 0.0
    availability = _availability()
    availability["sender_component"] = np.nan

    result = score_communication(
        availability,
        _downstream(),
        _functional(weights=weights),
    )

    assert result.table["comm_strength"].notna().all()
    assert set(result.table["status"]) == {CommunicationScoreStatus.OK.value}


def test_out_of_fold_inputs_require_tags_no_leakage_and_common_id() -> None:
    functional = _functional(status=ScoringFunctionalStatus.OUT_OF_FOLD)
    availability = _tagged(_availability(), functional)
    downstream = _tagged(_downstream(), functional)

    result = score_communication(availability, downstream, functional)

    assert set(result.table["scoring_function_id"]) == {functional.scoring_function_id}
    assert set(result.table["fold_id"]) == {"fold-1"}
    assert set(result.table["functional_status"]) == {"out_of_fold"}
    assert set(result.table["context"]) == {"control", "treated"}

    with pytest.raises(ValueError, match="must carry scoring_function_id"):
        score_communication(_availability(), downstream, functional)

    leaking = availability.copy()
    leaking.loc[0, "subject_id"] = "train-1"
    leaking_downstream = downstream.copy()
    leaking_downstream.loc[0, "subject_id"] = "train-1"
    with pytest.raises(ValueError, match="training/test subject overlap"):
        score_communication(leaking, leaking_downstream, functional)


def test_formal_common_functional_contract_rejects_context_specific_ids() -> None:
    functional = _functional(status=ScoringFunctionalStatus.OUT_OF_FOLD)
    result = score_communication(
        _tagged(_availability(), functional),
        _tagged(_downstream(), functional),
        functional,
    )
    inconsistent = result.table.copy()
    inconsistent.loc[inconsistent["context"] == "treated", "scoring_function_id"] = (
        "scoring_function_context_specific"
    )

    with pytest.raises(ValueError, match="different scoring_function_id"):
        validate_common_functional(inconsistent)
    with pytest.raises(ValueError, match="stable ID"):
        CommunicationScores(inconsistent, functional)


def test_output_and_contract_forbid_probability_and_test_fields() -> None:
    result = score_communication(_availability(), _downstream(), _functional())
    assert not any("probability" in column for column in result.table.columns)

    for forbidden in ("comm_probability", "p_value", "q_value"):
        invalid = result.table.assign(**{forbidden: 0.5})
        with pytest.raises(ValueError, match="forbidden columns"):
            CommunicationScores(invalid, result.functional)


def test_communication_score_contract_rejects_malformed_manual_tables() -> None:
    result = score_communication(_availability(), _downstream(), _functional())
    with pytest.raises(ValueError, match="table is missing"):
        CommunicationScores(result.table.drop(columns="mode"), result.functional)
    one_row = result.table.iloc[[0]].copy()
    with pytest.raises(ValueError, match="mode must be"):
        CommunicationScores(one_row.assign(mode="raw_lr_product"), result.functional)
    with pytest.raises(ValueError, match="status is not recognized"):
        CommunicationScores(one_row.assign(status="unknown"), result.functional)
    with pytest.raises(ValueError, match="numeric values"):
        CommunicationScores(
            one_row.assign(sender_component="not-a-number"), result.functional
        )
    with pytest.raises(ValueError, match="must not carry"):
        CommunicationScores(one_row.assign(reason_code="invented"), result.functional)
    with pytest.raises(ValueError, match="mode must be"):
        result.for_mode("probability")


def test_scoring_input_contracts_and_empty_exploratory_output() -> None:
    functional = _functional()
    with pytest.raises(TypeError, match="ScoringFunctional"):
        score_communication(  # type: ignore[arg-type]
            _availability(), _downstream(), object()
        )
    with pytest.raises(ValueError, match="missing columns"):
        score_communication(
            _availability().drop(columns="sender_component"),
            _downstream(),
            functional,
        )
    duplicated = pd.concat([_availability(), _availability().iloc[[0]]])
    with pytest.raises(ValueError, match="must be unique"):
        score_communication(duplicated, _downstream(), functional)
    mismatched = _availability().assign(scoring_function_id="wrong")
    with pytest.raises(ValueError, match="not generated"):
        score_communication(mismatched, _downstream(), functional)

    empty_availability = _availability().iloc[0:0].copy()
    empty = score_communication(empty_availability, _downstream(), functional)
    assert empty.table.empty
    assert set(empty.table.columns).issuperset(
        {"comm_strength", "scoring_function_id", "mode"}
    )


@pytest.mark.parametrize(
    ("table_name", "column"),
    [
        ("availability", "state_availability"),
        ("availability", "sender_component"),
        ("downstream", "downstream_activity"),
        ("downstream", "prior_quality"),
    ],
)
def test_components_outside_unit_interval_are_rejected(
    table_name: str, column: str
) -> None:
    availability = _availability()
    downstream = _downstream()
    target = availability if table_name == "availability" else downstream
    target.loc[0, column] = 1.01

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        score_communication(availability, downstream, _functional())


def test_unknown_interaction_is_rejected_by_frozen_universe() -> None:
    availability = _availability()
    availability.loc[0, "interaction_id"] = "not-frozen"

    with pytest.raises(ValueError, match="outside the frozen universe"):
        score_communication(availability, _downstream(), _functional())
