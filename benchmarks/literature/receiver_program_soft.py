"""Reliability-shrunk receiver-program evidence for benchmark cardinality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

MIN_CONCORDANCE_EDGES = 200


@dataclass(frozen=True)
class ReceiverProgramSoftFit:
    """Unsupervised cross-channel reliability and bounded evidence strength."""

    sign_concordance: float
    reliability: float
    maximum_alpha: float
    alpha_cap: float
    effective_alpha: float
    n_concordance_edges: int
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TwoSidedProgramCalibration:
    """Coverage-derived attenuation for contradictory program evidence."""

    positive_program_edges: int
    negative_program_edges: int
    nonzero_program_edges: int
    minority_sign_fraction: float
    required_two_sided_fraction: float
    contradiction_scale: float
    fit_status: str = "candidate_unreleased"
    formal_release_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_receiver_program_reliability(
    lr_statistic: Any,
    program_z: Any,
    *,
    maximum_alpha: float,
    alpha_cap: float = 1.5,
    minimum_edges: int = MIN_CONCORDANCE_EDGES,
) -> ReceiverProgramSoftFit:
    """Shrink program strength using LR/program sign agreement above chance."""

    lr = np.asarray(lr_statistic, dtype=float)
    program = np.asarray(program_z, dtype=float)
    if lr.ndim != 1 or program.ndim != 1 or lr.shape != program.shape:
        raise ValueError("lr_statistic and program_z must be equal-length vectors")
    if not np.isfinite(maximum_alpha) or maximum_alpha < 0.0:
        raise ValueError("maximum_alpha must be finite and nonnegative")
    if not np.isfinite(alpha_cap) or alpha_cap <= 0.0:
        raise ValueError("alpha_cap must be finite and positive")
    eligible = np.isfinite(lr) & np.isfinite(program) & (lr != 0.0) & (program != 0.0)
    n_edges = int(eligible.sum())
    if n_edges < minimum_edges:
        return ReceiverProgramSoftFit(
            sign_concordance=float("nan"),
            reliability=0.0,
            maximum_alpha=maximum_alpha,
            alpha_cap=alpha_cap,
            effective_alpha=0.0,
            n_concordance_edges=n_edges,
        )
    concordance = float(np.mean(np.sign(lr[eligible]) == np.sign(program[eligible])))
    reliability = float(np.clip(2.0 * (concordance - 0.5), 0.0, 1.0))
    effective_alpha = min(alpha_cap, maximum_alpha * reliability)
    return ReceiverProgramSoftFit(
        sign_concordance=concordance,
        reliability=reliability,
        maximum_alpha=maximum_alpha,
        alpha_cap=alpha_cap,
        effective_alpha=effective_alpha,
        n_concordance_edges=n_edges,
    )


def receiver_program_weights(
    direction: Any,
    program_z: Any,
    fit: ReceiverProgramSoftFit,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive weights and oriented evidence; missing evidence is neutral."""

    sign = np.asarray(direction, dtype=float)
    program = np.asarray(program_z, dtype=float)
    if sign.ndim != 1 or program.ndim != 1 or sign.shape != program.shape:
        raise ValueError("direction and program_z must be equal-length vectors")
    if not np.isin(sign[np.isfinite(sign)], (-1.0, 1.0)).all():
        raise ValueError("finite direction values must equal -1 or 1")
    if fit.formal_release_allowed or fit.fit_status != "candidate_unreleased":
        raise ValueError("benchmark receiver-program release contract was altered")
    available = np.isfinite(sign) & np.isfinite(program)
    evidence = np.zeros(len(sign), dtype=float)
    evidence[available] = 2.0 * norm.cdf(sign[available] * program[available]) - 1.0
    weights = np.exp(fit.effective_alpha * evidence)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise AssertionError("receiver-program weights must be finite and positive")
    return weights, evidence


def two_sided_calibrated_program_weights(
    direction: Any,
    program_z: Any,
    fit: ReceiverProgramSoftFit,
    *,
    required_two_sided_fraction: float,
) -> tuple[np.ndarray, np.ndarray, TwoSidedProgramCalibration]:
    """Attenuate contradiction penalties when program signs are one-sided."""

    if (
        not np.isfinite(required_two_sided_fraction)
        or required_two_sided_fraction < 0.0
        or required_two_sided_fraction > 0.5
    ):
        raise ValueError("required two-sided fraction must lie in [0, 0.5]")
    _, oriented_evidence = receiver_program_weights(direction, program_z, fit)
    program = np.asarray(program_z, dtype=float)
    positive = int(np.sum(np.isfinite(program) & (program > 0.0)))
    negative = int(np.sum(np.isfinite(program) & (program < 0.0)))
    nonzero = positive + negative
    minority_fraction = min(positive, negative) / nonzero if nonzero else 0.0
    if required_two_sided_fraction == 0.0:
        contradiction_scale = 1.0
    else:
        contradiction_scale = min(1.0, minority_fraction / required_two_sided_fraction)
    calibrated_evidence = np.where(
        oriented_evidence < 0.0,
        contradiction_scale * oriented_evidence,
        oriented_evidence,
    )
    weights = np.exp(fit.effective_alpha * calibrated_evidence)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise AssertionError("calibrated program weights must be finite and positive")
    calibration = TwoSidedProgramCalibration(
        positive_program_edges=positive,
        negative_program_edges=negative,
        nonzero_program_edges=nonzero,
        minority_sign_fraction=minority_fraction,
        required_two_sided_fraction=required_two_sided_fraction,
        contradiction_scale=contradiction_scale,
    )
    return weights, calibrated_evidence, calibration


__all__ = [
    "MIN_CONCORDANCE_EDGES",
    "ReceiverProgramSoftFit",
    "TwoSidedProgramCalibration",
    "fit_receiver_program_reliability",
    "receiver_program_weights",
    "two_sided_calibrated_program_weights",
]
