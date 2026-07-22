"""Bounded convex blending of soft and gated receiver-program evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundedProgramBlendFit:
    """Normalization and blend diagnostics for a benchmark-only head."""

    blend_fraction: float
    call_count: int
    soft_call_mean: float
    gate_call_mean: float
    blended_call_mean: float
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def bounded_program_blend_weights(
    soft_weights: Any,
    gate_weights: Any,
    call_mask: Any,
    *,
    blend_fraction: float,
) -> tuple[np.ndarray, BoundedProgramBlendFit]:
    """Return a mean-preserving convex blend on the frozen LR call set."""

    soft = np.asarray(soft_weights, dtype=float)
    gate = np.asarray(gate_weights, dtype=float)
    calls = np.asarray(call_mask)
    if soft.ndim != 1 or gate.shape != soft.shape or calls.shape != soft.shape:
        raise ValueError("soft, gate, and call mask must be equal-length vectors")
    if not np.isfinite(soft).all() or not np.isfinite(gate).all():
        raise ValueError("program weights must be finite")
    if (soft <= 0.0).any() or (gate < 0.0).any():
        raise ValueError("soft weights must be positive and gate weights nonnegative")
    if not np.isfinite(blend_fraction) or not 0.0 <= blend_fraction <= 1.0:
        raise ValueError("blend_fraction must lie in [0, 1]")
    called = calls.astype(bool, copy=False)
    call_count = int(called.sum())
    if call_count == 0:
        fit = BoundedProgramBlendFit(
            blend_fraction=blend_fraction,
            call_count=0,
            soft_call_mean=float("nan"),
            gate_call_mean=float("nan"),
            blended_call_mean=float("nan"),
        )
        return soft.copy(), fit
    soft_mean = float(soft[called].mean())
    gate_mean = float(gate[called].mean())
    if soft_mean <= 0.0 or gate_mean <= 0.0:
        raise ValueError("each program expert must retain positive call-set mass")
    normalized_soft = soft / soft_mean
    normalized_gate = gate / gate_mean
    blended = (
        (1.0 - blend_fraction) * normalized_soft
        + blend_fraction * normalized_gate
    )
    fit = BoundedProgramBlendFit(
        blend_fraction=blend_fraction,
        call_count=call_count,
        soft_call_mean=soft_mean,
        gate_call_mean=gate_mean,
        blended_call_mean=float(blended[called].mean()),
    )
    return blended, fit
