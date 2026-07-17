from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import (
    DirectionalContrastPairSpec,
    DirectionalPairReconstruction,
    DirectionalResponsePair,
    build_directional_response_pair,
)
from crychic.core import ContractError
from crychic.design import ContrastSpec, balanced_contrast


def _contrasts() -> tuple[ContrastSpec, ContrastSpec]:
    forward = balanced_contrast(
        ("stim",),
        ("control",),
        name="stim_vs_control",
        family="treatment",
    )
    reverse = balanced_contrast(
        ("control",),
        ("stim",),
        name="control_vs_stim",
        family="treatment",
    )
    return forward, reverse


def _pair() -> DirectionalResponsePair:
    forward, reverse = _contrasts()
    return build_directional_response_pair(
        ("up", "down", "flat"),
        forward,
        reverse,
        np.asarray([3.0, -2.0, 0.0]),
        np.asarray([-3.0, 2.0, 0.0]),
        forward_response_id="response-forward",
        reverse_response_id="response-reverse",
    )


def test_contrast_pair_spec_binds_explicit_order_and_stable_identity() -> None:
    forward, reverse = _contrasts()
    first = DirectionalContrastPairSpec(
        forward_contrast=forward,
        reverse_contrast=reverse,
    )
    repeated = DirectionalContrastPairSpec(
        forward_contrast=forward,
        reverse_contrast=reverse,
    )
    opposite_order = DirectionalContrastPairSpec(
        forward_contrast=reverse,
        reverse_contrast=forward,
    )

    assert first.pair_spec_id == repeated.pair_spec_id
    assert first.pair_spec_id != opposite_order.pair_spec_id
    assert first.forward_contrast_id == opposite_order.reverse_contrast_id
    assert first.reverse_contrast_id == opposite_order.forward_contrast_id
    assert first.forward_channel_name == "increased_activation_compatible"
    assert first.reverse_channel_name == "reduced_activation_compatible"
    assert first.supports_active_inhibition_claim is False
    assert first.combination_rule == "independent_contrast_views_not_additive_v1"
    assert first.schema_version == "1.0.0"
    assert first.to_dict()["forward_contrast"] == forward.to_dict()
    assert first.to_dict()["reverse_contrast"] == reverse.to_dict()


def test_contrast_pair_spec_identity_is_invariant_to_weight_input_order() -> None:
    first = DirectionalContrastPairSpec(
        forward_contrast=ContrastSpec(
            name="stim_vs_control",
            weights={"stim": 1.0, "control": -1.0},
            family="treatment",
            mode="balanced",
        ),
        reverse_contrast=ContrastSpec(
            name="control_vs_stim",
            weights={"control": 1.0, "stim": -1.0},
            family="treatment",
            mode="balanced",
        ),
    )
    second = DirectionalContrastPairSpec(
        forward_contrast=ContrastSpec(
            name="stim_vs_control",
            weights={"control": -1.0, "stim": 1.0},
            family="treatment",
            mode="balanced",
        ),
        reverse_contrast=ContrastSpec(
            name="control_vs_stim",
            weights={"stim": -1.0, "control": 1.0},
            family="treatment",
            mode="balanced",
        ),
    )

    assert first.pair_spec_id == second.pair_spec_id
    assert first.to_dict() == second.to_dict()


def test_contrast_pair_spec_rejects_inexact_reverse_and_detects_tampering() -> None:
    forward, reverse = _contrasts()
    nearly_reversed = ContrastSpec(
        name="nearly_control_vs_stim",
        weights={"control": 1.0 - 5e-13, "stim": -(1.0 - 5e-13)},
        family="treatment",
        mode="balanced",
    )
    with pytest.raises(ContractError) as reverse_error:
        DirectionalContrastPairSpec(
            forward_contrast=forward,
            reverse_contrast=nearly_reversed,
        )
    assert reverse_error.value.details.code == "invalid_directional_contrast_pair"

    spec = DirectionalContrastPairSpec(
        forward_contrast=forward,
        reverse_contrast=reverse,
    )
    object.__setattr__(spec, "combination_rule", "additive")
    with pytest.raises(ContractError) as tamper_error:
        spec.to_dict()
    assert tamper_error.value.details.code == (
        "directional_contrast_pair_spec_integrity_violation"
    )


def test_pair_exposes_only_nonnegative_solver_channels_with_stable_identity() -> None:
    source = np.asarray([3.0, -2.0, 0.0])
    forward, reverse = _contrasts()
    first = build_directional_response_pair(
        ("up", "down", "flat"),
        forward,
        reverse,
        source,
        -source,
        forward_response_id="response-forward",
        reverse_response_id="response-reverse",
    )
    second = _pair()
    distinct_provenance = build_directional_response_pair(
        ("up", "down", "flat"),
        forward,
        reverse,
        np.asarray([3.0, -2.0, 0.0]),
        np.asarray([-3.0, 2.0, 0.0]),
        forward_response_id="response-forward",
        reverse_response_id="response-reverse-rerun",
    )
    source[:] = 99.0

    np.testing.assert_array_equal(first.forward_signed_response, [3.0, -2.0, 0.0])
    np.testing.assert_array_equal(first.forward_solver_response, [3.0, 0.0, 0.0])
    np.testing.assert_array_equal(first.reverse_solver_response, [0.0, 2.0, 0.0])
    assert first.response_pair_id == second.response_pair_id
    assert first.response_pair_id != distinct_provenance.response_pair_id
    assert first.forward_channel_id != first.reverse_channel_id
    assert first.forward_contrast_id != first.reverse_contrast_id
    for values in (
        first.forward_signed_response,
        first.reverse_signed_response,
        first.forward_solver_response,
        first.reverse_solver_response,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_forward_and_reverse_reconstruct_independent_signed_views() -> None:
    pair = _pair()
    result = pair.reconstruct(
        np.asarray([2.5, 0.25, 0.0]),
        np.asarray([0.1, 1.5, 0.0]),
    )
    repeated = pair.reconstruct(
        np.asarray([2.5, 0.25, 0.0]),
        np.asarray([0.1, 1.5, 0.0]),
    )

    np.testing.assert_allclose(
        result.forward_prediction + result.forward_signed_residual,
        pair.forward_signed_response,
    )
    np.testing.assert_allclose(
        result.reverse_prediction + result.reverse_signed_residual,
        pair.reverse_signed_response,
    )
    np.testing.assert_allclose(
        result.forward_reconstructed_response,
        pair.forward_signed_response,
    )
    np.testing.assert_allclose(
        result.reverse_reconstructed_on_forward_scale,
        pair.forward_signed_response,
    )
    np.testing.assert_array_equal(
        result.reverse_prediction_on_forward_scale,
        [-0.1, -1.5, 0.0],
    )
    assert result.reconstruction_id == repeated.reconstruction_id
    assert result.to_dict()["combination_rule"] == (
        "independent_contrast_views_not_additive_v1"
    )


def test_negative_forward_effect_remains_signed_residual_not_inhibition() -> None:
    pair = _pair()
    forward_only = pair.reconstruct(
        np.asarray([2.5, 0.0, 0.0]),
        np.asarray([0.0, 1.5, 0.0]),
    )

    assert forward_only.forward_signed_residual[1] == pytest.approx(-2.0)
    assert pair.reverse_biological_claim == (
        "attenuation_of_activation_from_explicit_reverse_contrast"
    )
    assert "inhibition" not in pair.reverse_biological_claim
    assert pair.supports_active_inhibition_claim is False
    assert forward_only.supports_active_inhibition_claim is False


def test_pair_rejects_implicit_or_misaligned_reverse_channel() -> None:
    forward, reverse = _contrasts()
    common = {
        "feature_ids": ("G1", "G2"),
        "forward_contrast": forward,
        "reverse_contrast": reverse,
        "forward_signed_response": np.asarray([1.0, -1.0]),
        "reverse_signed_response": np.asarray([-1.0, 1.0]),
        "forward_response_id": "response-forward",
        "reverse_response_id": "response-reverse",
    }

    with pytest.raises(ContractError) as provenance_error:
        build_directional_response_pair(
            **{**common, "reverse_response_id": "response-forward"}
        )
    assert provenance_error.value.details.code == (
        "invalid_directional_response_provenance"
    )

    with pytest.raises(ContractError) as contrast_error:
        build_directional_response_pair(**{**common, "reverse_contrast": forward})
    assert contrast_error.value.details.code == "invalid_directional_contrast_pair"

    with pytest.raises(ContractError) as response_error:
        build_directional_response_pair(
            **{
                **common,
                "reverse_signed_response": np.asarray([-1.0, 0.5]),
            }
        )
    assert response_error.value.details.code == "invalid_directional_response_pair"


def test_pair_rejects_signed_prior_and_invalid_channel_predictions() -> None:
    forward, reverse = _contrasts()
    with pytest.raises(ContractError) as prior_error:
        build_directional_response_pair(
            ("G1",),
            forward,
            reverse,
            np.asarray([1.0]),
            np.asarray([-1.0]),
            forward_response_id="response-forward",
            reverse_response_id="response-reverse",
            prior_direction=-1,
        )
    assert prior_error.value.details.code == "unsupported_signed_prior_semantics"

    with pytest.raises(ContractError) as typed_prior_error:
        build_directional_response_pair(
            ("G1",),
            forward,
            reverse,
            np.asarray([1.0]),
            np.asarray([-1.0]),
            forward_response_id="response-forward",
            reverse_response_id="response-reverse",
            prior_direction=1.0,
        )
    assert typed_prior_error.value.details.code == "unsupported_signed_prior_semantics"

    with pytest.raises(TypeError, match="sequence"):
        build_directional_response_pair(
            "G1",
            forward,
            reverse,
            np.asarray([1.0, -1.0]),
            np.asarray([-1.0, 1.0]),
            forward_response_id="response-forward",
            reverse_response_id="response-reverse",
        )

    pair = _pair()
    with pytest.raises(ContractError) as prediction_error:
        pair.reconstruct(
            np.asarray([1.0, -0.1, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
        )
    assert prediction_error.value.details.code == "invalid_directional_prediction"

    with pytest.raises(ContractError) as alignment_error:
        pair.reconstruct(np.asarray([1.0]), np.asarray([1.0]))
    assert alignment_error.value.details.code == (
        "invalid_directional_feature_alignment"
    )


def test_reconstruction_contract_rejects_forged_signed_residual() -> None:
    pair = _pair()
    valid = pair.reconstruct(
        np.asarray([2.5, 0.0, 0.0]),
        np.asarray([0.0, 1.5, 0.0]),
    )

    with pytest.raises(ContractError) as error:
        DirectionalPairReconstruction(
            response_pair_id=valid.response_pair_id,
            forward_channel_id=valid.forward_channel_id,
            reverse_channel_id=valid.reverse_channel_id,
            feature_ids=valid.feature_ids,
            observed_forward_response=valid.observed_forward_response,
            observed_reverse_response=valid.observed_reverse_response,
            forward_prediction=valid.forward_prediction,
            reverse_prediction=valid.reverse_prediction,
            forward_signed_residual=np.zeros(3),
            reverse_signed_residual=valid.reverse_signed_residual,
        )
    assert error.value.details.code == "invalid_directional_reconstruction"
