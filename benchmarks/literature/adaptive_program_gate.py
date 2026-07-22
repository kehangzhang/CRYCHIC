"""Outcome-free mirror-tail calibration for receiver-program benchmark gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveProgramGateFit:
    """Frozen diagnostics for a missing-neutral benchmark ranking gate."""

    relative_fdp_reduction: float
    threshold: float
    gate_mode: str
    eligible_call_count: int
    positive_call_count: int
    negative_call_count: int
    sign_concordance: float
    baseline_mirror_fdp: float
    target_mirror_fdp: float
    selected_mirror_fdp: float
    selected_tail_concordance_lower_bound: float
    retained_positive_calls: int
    rejected_negative_calls: int
    minimum_calls: int
    minimum_sign_concordance: float
    minimum_positive_tail_count: int
    minimum_positive_tail_fraction: float
    tail_confidence_z: float
    minimum_tail_concordance_margin: float
    validation_fold_count: int
    minimum_validation_tail_count: int
    minimum_validation_concordance_gain: float
    selected_minimum_validation_concordance_gain: float
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector(values: Any, *, field: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    return result


def _no_gate_fit(
    *,
    relative_fdp_reduction: float,
    eligible_count: int,
    positive_count: int,
    negative_count: int,
    concordance: float,
    baseline_fdp: float,
    minimum_calls: int,
    minimum_sign_concordance: float,
    minimum_positive_tail_count: int,
    minimum_positive_tail_fraction: float,
    tail_confidence_z: float,
    minimum_tail_concordance_margin: float,
    validation_fold_count: int,
    minimum_validation_tail_count: int,
    minimum_validation_concordance_gain: float,
) -> AdaptiveProgramGateFit:
    return AdaptiveProgramGateFit(
        relative_fdp_reduction=relative_fdp_reduction,
        threshold=float("nan"),
        gate_mode="no_gate",
        eligible_call_count=eligible_count,
        positive_call_count=positive_count,
        negative_call_count=negative_count,
        sign_concordance=concordance,
        baseline_mirror_fdp=baseline_fdp,
        target_mirror_fdp=float("nan"),
        selected_mirror_fdp=float("nan"),
        selected_tail_concordance_lower_bound=float("nan"),
        retained_positive_calls=positive_count,
        rejected_negative_calls=0,
        minimum_calls=minimum_calls,
        minimum_sign_concordance=minimum_sign_concordance,
        minimum_positive_tail_count=minimum_positive_tail_count,
        minimum_positive_tail_fraction=minimum_positive_tail_fraction,
        tail_confidence_z=tail_confidence_z,
        minimum_tail_concordance_margin=minimum_tail_concordance_margin,
        validation_fold_count=validation_fold_count,
        minimum_validation_tail_count=minimum_validation_tail_count,
        minimum_validation_concordance_gain=minimum_validation_concordance_gain,
        selected_minimum_validation_concordance_gain=float("nan"),
    )


def fit_adaptive_program_gate(
    direction: Any,
    program_z: Any,
    call_mask: Any,
    *,
    relative_fdp_reduction: float,
    minimum_calls: int = 100,
    minimum_sign_concordance: float = 0.55,
    minimum_positive_tail_count: int = 20,
    minimum_positive_tail_fraction: float = 0.05,
    tail_confidence_z: float = 1.6448536269514722,
    minimum_tail_concordance_margin: float = 0.0,
    validation_fold: Any | None = None,
    minimum_validation_tail_count: int = 10,
    minimum_validation_concordance_gain: float = 0.0,
) -> AdaptiveProgramGateFit:
    """Calibrate a gate from positive versus mirrored negative program tails.

    Calibration uses only the already frozen LR call set. If program direction
    is unreliable, the fit leaves every call unchanged. If direction is
    reliable but magnitude adds no tail separation, it removes only
    contradictory evidence. A stronger threshold opens only when the mirrored
    tail ratio improves by the requested relative factor.
    """

    sign = _vector(direction, field="direction")
    program = _vector(program_z, field="program_z")
    calls = np.asarray(call_mask)
    if sign.shape != program.shape or calls.shape != sign.shape:
        raise ValueError("direction, program_z, and call_mask must have equal length")
    if calls.ndim != 1:
        raise ValueError("call_mask must be one-dimensional")
    if not np.isin(sign[np.isfinite(sign)], (-1.0, 1.0)).all():
        raise ValueError("finite direction values must equal -1 or 1")
    if not np.isfinite(relative_fdp_reduction) or not (
        0.0 < relative_fdp_reduction <= 1.0
    ):
        raise ValueError("relative_fdp_reduction must lie in (0, 1]")
    if minimum_calls < 1 or minimum_positive_tail_count < 1:
        raise ValueError("minimum call and tail counts must be positive")
    if not 0.5 <= minimum_sign_concordance <= 1.0:
        raise ValueError("minimum_sign_concordance must lie in [0.5, 1]")
    if not 0.0 <= minimum_positive_tail_fraction <= 1.0:
        raise ValueError("minimum_positive_tail_fraction must lie in [0, 1]")
    if not np.isfinite(tail_confidence_z) or tail_confidence_z < 0.0:
        raise ValueError("tail_confidence_z must be finite and nonnegative")
    if (
        not np.isfinite(minimum_tail_concordance_margin)
        or minimum_tail_concordance_margin < 0.0
        or minimum_tail_concordance_margin > 0.5
    ):
        raise ValueError(
            "minimum_tail_concordance_margin must be finite and lie in [0, 0.5]"
        )
    if minimum_validation_tail_count < 1:
        raise ValueError("minimum_validation_tail_count must be positive")
    if (
        not np.isfinite(minimum_validation_concordance_gain)
        or minimum_validation_concordance_gain < 0.0
        or minimum_validation_concordance_gain > 0.5
    ):
        raise ValueError(
            "minimum_validation_concordance_gain must be finite and lie in [0, 0.5]"
        )
    if validation_fold is None:
        folds = np.zeros(len(sign), dtype=int)
        validation_fold_count = 0
    else:
        folds = np.asarray(validation_fold)
        if folds.ndim != 1 or folds.shape != sign.shape:
            raise ValueError("validation_fold must be one-dimensional and match inputs")
        if not np.issubdtype(folds.dtype, np.integer):
            raise ValueError("validation_fold must contain integer fold identifiers")
        validation_fold_count = len(np.unique(folds))
        if validation_fold_count < 2:
            raise ValueError("validation_fold must contain at least two folds")

    called = calls.astype(bool, copy=False)
    eligible = called & np.isfinite(sign) & np.isfinite(program) & (program != 0.0)
    oriented = sign[eligible] * program[eligible]
    eligible_folds = folds[eligible]
    positive_count = int(np.sum(oriented > 0.0))
    negative_count = int(np.sum(oriented < 0.0))
    eligible_count = positive_count + negative_count
    concordance = (
        positive_count / eligible_count if eligible_count > 0 else float("nan")
    )
    baseline_fdp = (
        (1.0 + negative_count) / positive_count
        if positive_count > 0
        else float("inf")
    )
    no_gate = {
        "relative_fdp_reduction": relative_fdp_reduction,
        "eligible_count": eligible_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "concordance": concordance,
        "baseline_fdp": baseline_fdp,
        "minimum_calls": minimum_calls,
        "minimum_sign_concordance": minimum_sign_concordance,
        "minimum_positive_tail_count": minimum_positive_tail_count,
        "minimum_positive_tail_fraction": minimum_positive_tail_fraction,
        "tail_confidence_z": tail_confidence_z,
        "minimum_tail_concordance_margin": minimum_tail_concordance_margin,
        "validation_fold_count": validation_fold_count,
        "minimum_validation_tail_count": minimum_validation_tail_count,
        "minimum_validation_concordance_gain": minimum_validation_concordance_gain,
    }
    if (
        eligible_count < minimum_calls
        or positive_count == 0
        or not np.isfinite(concordance)
        or concordance < minimum_sign_concordance
    ):
        return _no_gate_fit(**no_gate)

    target_fdp = relative_fdp_reduction * baseline_fdp
    minimum_tail = max(
        minimum_positive_tail_count,
        int(np.ceil(minimum_positive_tail_fraction * positive_count)),
    )
    selected_threshold = 0.0
    selected_fdp = baseline_fdp
    retained_positive = positive_count
    selected_lower_bound = float("nan")
    selected_validation_gain = float("nan")
    gate_mode = "sign_only"
    if relative_fdp_reduction < 1.0:
        thresholds = np.unique(np.abs(oriented))
        thresholds = thresholds[thresholds > 0.0]
        for threshold in thresholds:
            positive_tail = int(np.sum(oriented >= threshold))
            if positive_tail < minimum_tail:
                break
            negative_tail = int(np.sum(oriented <= -threshold))
            mirror_fdp = (1.0 + negative_tail) / positive_tail
            tail_total = positive_tail + negative_tail
            tail_fraction = positive_tail / tail_total
            denominator = 1.0 + tail_confidence_z**2 / tail_total
            center = (
                tail_fraction + tail_confidence_z**2 / (2.0 * tail_total)
            ) / denominator
            margin = (
                tail_confidence_z
                * np.sqrt(
                    tail_fraction * (1.0 - tail_fraction) / tail_total
                    + tail_confidence_z**2 / (4.0 * tail_total**2)
                )
                / denominator
            )
            lower_bound = center - margin
            validation_gains: list[float] = []
            validation_pass = True
            if validation_fold_count:
                for fold in np.unique(eligible_folds):
                    fold_values = oriented[eligible_folds == fold]
                    fold_positive = int(np.sum(fold_values > 0.0))
                    fold_negative = int(np.sum(fold_values < 0.0))
                    fold_total = fold_positive + fold_negative
                    tail_positive = int(np.sum(fold_values >= threshold))
                    tail_negative = int(np.sum(fold_values <= -threshold))
                    tail_total = tail_positive + tail_negative
                    if (
                        fold_total == 0
                        or tail_positive < minimum_validation_tail_count
                        or tail_total == 0
                    ):
                        validation_pass = False
                        break
                    fold_concordance = fold_positive / fold_total
                    tail_concordance = tail_positive / tail_total
                    validation_gains.append(tail_concordance - fold_concordance)
                    if (
                        tail_concordance
                        < fold_concordance + minimum_validation_concordance_gain
                    ):
                        validation_pass = False
                        break
            if (
                mirror_fdp <= target_fdp
                and lower_bound > concordance + minimum_tail_concordance_margin
                and validation_pass
            ):
                selected_threshold = float(threshold)
                selected_fdp = float(mirror_fdp)
                retained_positive = positive_tail
                selected_lower_bound = float(lower_bound)
                selected_validation_gain = (
                    float(min(validation_gains))
                    if validation_gains
                    else float("nan")
                )
                gate_mode = "enriched_tail"
                break

    return AdaptiveProgramGateFit(
        relative_fdp_reduction=relative_fdp_reduction,
        threshold=selected_threshold,
        gate_mode=gate_mode,
        eligible_call_count=eligible_count,
        positive_call_count=positive_count,
        negative_call_count=negative_count,
        sign_concordance=concordance,
        baseline_mirror_fdp=baseline_fdp,
        target_mirror_fdp=target_fdp,
        selected_mirror_fdp=selected_fdp,
        selected_tail_concordance_lower_bound=selected_lower_bound,
        retained_positive_calls=retained_positive,
        rejected_negative_calls=negative_count,
        minimum_calls=minimum_calls,
        minimum_sign_concordance=minimum_sign_concordance,
        minimum_positive_tail_count=minimum_positive_tail_count,
        minimum_positive_tail_fraction=minimum_positive_tail_fraction,
        tail_confidence_z=tail_confidence_z,
        minimum_tail_concordance_margin=minimum_tail_concordance_margin,
        validation_fold_count=validation_fold_count,
        minimum_validation_tail_count=minimum_validation_tail_count,
        minimum_validation_concordance_gain=minimum_validation_concordance_gain,
        selected_minimum_validation_concordance_gain=selected_validation_gain,
    )


def adaptive_program_gate_weights(
    direction: Any,
    program_z: Any,
    fit: AdaptiveProgramGateFit,
) -> np.ndarray:
    """Apply a fitted binary gate while retaining missing program evidence."""

    sign = _vector(direction, field="direction")
    program = _vector(program_z, field="program_z")
    if sign.shape != program.shape:
        raise ValueError("direction and program_z must have equal length")
    if fit.formal_release_allowed or fit.fit_status != "candidate_unreleased":
        raise ValueError("benchmark adaptive-program release contract was altered")
    weights = np.ones(len(sign), dtype=float)
    if fit.gate_mode == "no_gate":
        return weights
    if fit.gate_mode not in {"sign_only", "enriched_tail"}:
        raise ValueError(f"unknown adaptive program gate mode: {fit.gate_mode}")
    available = np.isfinite(sign) & np.isfinite(program)
    oriented = sign[available] * program[available]
    if fit.gate_mode == "sign_only":
        weights[available] = (oriented > 0.0).astype(float)
    else:
        weights[available] = (oriented >= fit.threshold).astype(float)
    return weights
