from __future__ import annotations

import pytest

from crychic.scoring import (
    SIGNED_PROGRAM_CONCORDANCE_FORMULA,
    SIGNED_PROGRAM_CONCORDANCE_VERSION,
    signed_geometric_program_concordance,
)


def test_signed_program_concordance_preserves_activation_and_inhibition() -> None:
    assert signed_geometric_program_concordance(
        0.16, 0.09, expected_program_direction=1
    ) == pytest.approx(0.12)
    assert signed_geometric_program_concordance(
        -0.16, -0.09, expected_program_direction=1
    ) == pytest.approx(-0.12)
    assert signed_geometric_program_concordance(
        0.16, -0.09, expected_program_direction=-1
    ) == pytest.approx(0.12)
    assert signed_geometric_program_concordance(
        -0.16, 0.09, expected_program_direction=-1
    ) == pytest.approx(-0.12)


def test_signed_program_concordance_maps_discordance_and_zero_to_zero() -> None:
    assert (
        signed_geometric_program_concordance(
            -0.16, 0.09, expected_program_direction=1
        )
        == 0.0
    )
    assert (
        signed_geometric_program_concordance(
            0.16, 0.0, expected_program_direction=1
        )
        == 0.0
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_signed_program_concordance_rejects_nonfinite_inputs(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        signed_geometric_program_concordance(
            value, 0.1, expected_program_direction=1
        )
    with pytest.raises(ValueError, match="must be finite"):
        signed_geometric_program_concordance(
            0.1, value, expected_program_direction=1
        )


def test_signed_program_concordance_exports_frozen_identity() -> None:
    assert SIGNED_PROGRAM_CONCORDANCE_VERSION == (
        "signed_geometric_program_concordance_m1_v1"
    )
    assert "aligned_program_effect" in SIGNED_PROGRAM_CONCORDANCE_FORMULA
    with pytest.raises(ValueError, match="-1 or 1"):
        signed_geometric_program_concordance(
            0.1, 0.1, expected_program_direction=0
        )
