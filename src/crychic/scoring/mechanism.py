"""Signed descriptive mechanism-support heads kept separate from activity."""

from __future__ import annotations

import math

SIGNED_PROGRAM_CONCORDANCE_VERSION = "signed_geometric_program_concordance_m1_v1"
SIGNED_PROGRAM_CONCORDANCE_FORMULA = (
    "sign(lr_effect)*sqrt(abs(lr_effect)*abs(aligned_program_effect))_"
    "when_same_direction_else_zero"
)


def signed_geometric_program_concordance(
    lr_effect: float,
    program_effect: float,
    *,
    expected_program_direction: int,
) -> float:
    """Return signed descriptive support when LR and frozen program align."""

    lr = float(lr_effect)
    program = float(program_effect)
    if not math.isfinite(lr) or not math.isfinite(program):
        raise ValueError("signed program-concordance inputs must be finite")
    if expected_program_direction not in {-1, 1}:
        raise ValueError("expected_program_direction must be -1 or 1")
    aligned = expected_program_direction * program
    if (
        lr == 0.0
        or aligned == 0.0
        or math.copysign(1.0, lr) != math.copysign(1.0, aligned)
    ):
        return 0.0
    return float(math.copysign(math.sqrt(abs(lr) * abs(aligned)), lr))


__all__ = [
    "SIGNED_PROGRAM_CONCORDANCE_FORMULA",
    "SIGNED_PROGRAM_CONCORDANCE_VERSION",
    "signed_geometric_program_concordance",
]
