from __future__ import annotations

import numpy as np
import pytest

from crychic.attribution import ResponseDirection, directional_response_channel


def test_increase_channel_preserves_negative_values_in_signed_residual() -> None:
    channel = directional_response_channel(
        ("up", "down", "flat"),
        np.asarray([2.0, -3.0, 0.0]),
        direction=ResponseDirection.INCREASE,
    )

    np.testing.assert_array_equal(channel.compatible_response, [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(channel.unmatched_signed_response, [0.0, -3.0, 0.0])
    np.testing.assert_array_equal(
        channel.full_signed_residual(np.asarray([1.5, 0.0, 0.0])),
        [0.5, -3.0, 0.0],
    )
    assert channel.biological_claim == "direction_compatible_activation"


def test_reverse_channel_maps_attenuation_back_to_negative_signed_effect() -> None:
    channel = directional_response_channel(
        ("up", "down"),
        np.asarray([2.0, -3.0]),
        direction=ResponseDirection.ATTENUATION_REVERSE_CONTRAST,
    )

    np.testing.assert_array_equal(channel.compatible_response, [0.0, 3.0])
    np.testing.assert_array_equal(channel.prediction_to_signed([0.0, 2.0]), [0, -2])
    np.testing.assert_array_equal(
        channel.full_signed_residual(np.asarray([0.0, 2.0])), [2.0, -1.0]
    )
    assert channel.biological_claim == "reduced_activation_not_active_inhibition"
    assert "inhibition" in channel.biological_claim


def test_directional_channel_is_deterministic_and_rejects_signed_predictions() -> None:
    first = directional_response_channel(
        ("G1", "G2"),
        np.asarray([1.0, -1.0]),
        direction="increase",
    )
    second = directional_response_channel(
        ("G1", "G2"),
        np.asarray([1.0, -1.0]),
        direction=ResponseDirection.INCREASE,
    )

    assert first.channel_id == second.channel_id
    with pytest.raises(ValueError, match="non-negative"):
        first.prediction_to_signed(np.asarray([1.0, -0.1]))


def test_directional_contract_rejects_forged_reconstruction() -> None:
    channel = directional_response_channel(
        ("G1",), np.asarray([1.0]), direction="increase"
    )

    with pytest.raises(ValueError, match="reconstruct"):
        type(channel)(
            feature_ids=channel.feature_ids,
            signed_response=channel.signed_response,
            compatible_response=channel.compatible_response,
            unmatched_signed_response=np.asarray([1.0]),
            direction=channel.direction,
        )
