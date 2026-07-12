"""Explicit direction-compatible channels for non-negative attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from crychic.core import stable_id


class ResponseDirection(StrEnum):
    """Reviewed semantics for a non-negative molecular prior."""

    INCREASE = "increase"
    ATTENUATION_REVERSE_CONTRAST = "attenuation_reverse_contrast"

    @property
    def multiplier(self) -> float:
        return 1.0 if self is ResponseDirection.INCREASE else -1.0

    @property
    def biological_claim(self) -> str:
        return (
            "direction_compatible_activation"
            if self is ResponseDirection.INCREASE
            else "reduced_activation_not_active_inhibition"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalResponseChannel:
    """Positive solver response plus signed residual reconstruction contract."""

    feature_ids: tuple[str, ...]
    signed_response: np.ndarray
    compatible_response: np.ndarray
    unmatched_signed_response: np.ndarray
    direction: ResponseDirection
    channel_id: str = field(init=False)

    def __post_init__(self) -> None:
        features = tuple(self.feature_ids)
        if not features or any(not value for value in features):
            raise ValueError("feature_ids must contain non-empty strings")
        if len(set(features)) != len(features):
            raise ValueError("feature_ids must be unique")
        direction = ResponseDirection(self.direction)
        arrays: list[np.ndarray] = []
        for field_name in (
            "signed_response",
            "compatible_response",
            "unmatched_signed_response",
        ):
            array: np.ndarray = np.asarray(
                getattr(self, field_name), dtype=np.float64
            ).copy()
            if array.shape != (len(features),) or np.any(~np.isfinite(array)):
                raise ValueError(
                    f"{field_name} must be a finite vector aligned to feature_ids"
                )
            array.setflags(write=False)
            arrays.append(array)
        signed, compatible, unmatched = arrays
        if np.any(compatible < 0):
            raise ValueError("compatible_response must be non-negative")
        reconstructed = direction.multiplier * compatible + unmatched
        if not np.allclose(reconstructed, signed, rtol=0.0, atol=1e-12):
            raise ValueError("directional response does not reconstruct signed input")
        expected = np.maximum(direction.multiplier * signed, 0.0)
        if not np.allclose(expected, compatible, rtol=0.0, atol=1e-12):
            raise ValueError("compatible_response does not match direction semantics")
        payload = {
            "biological_claim": direction.biological_claim,
            "compatible_response": compatible.tolist(),
            "direction": direction.value,
            "feature_ids": list(features),
            "signed_response": signed.tolist(),
        }
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "signed_response", signed)
        object.__setattr__(self, "compatible_response", compatible)
        object.__setattr__(self, "unmatched_signed_response", unmatched)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self, "channel_id", stable_id("directional_response_channel", payload)
        )

    @property
    def biological_claim(self) -> str:
        return self.direction.biological_claim

    def prediction_to_signed(self, predicted_compatible: np.ndarray) -> np.ndarray:
        """Map a non-negative solver prediction back to the signed response scale."""

        predicted: np.ndarray = np.asarray(
            predicted_compatible, dtype=np.float64
        ).copy()
        if predicted.shape != self.compatible_response.shape:
            raise ValueError("predicted response must align to the channel features")
        if np.any(~np.isfinite(predicted)) or np.any(predicted < 0):
            raise ValueError("compatible prediction must be finite and non-negative")
        signed_prediction: np.ndarray = self.direction.multiplier * predicted
        return signed_prediction

    def full_signed_residual(self, predicted_compatible: np.ndarray) -> np.ndarray:
        """Retain unmatched signs when calculating the complete signed residual."""

        residual: np.ndarray = self.signed_response - self.prediction_to_signed(
            predicted_compatible
        )
        return residual

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "direction": self.direction.value,
            "direction_multiplier": self.direction.multiplier,
            "biological_claim": self.biological_claim,
            "feature_ids": list(self.feature_ids),
        }


def directional_response_channel(
    feature_ids: tuple[str, ...],
    signed_response: np.ndarray,
    *,
    direction: ResponseDirection | str,
) -> DirectionalResponseChannel:
    """Build one reviewed positive or reverse-contrast solver channel."""

    selected = ResponseDirection(direction)
    signed = np.asarray(signed_response, dtype=np.float64)
    if signed.shape != (len(feature_ids),) or np.any(~np.isfinite(signed)):
        raise ValueError("signed_response must be finite and align to feature_ids")
    compatible = np.maximum(selected.multiplier * signed, 0.0)
    unmatched = signed - selected.multiplier * compatible
    return DirectionalResponseChannel(
        feature_ids=feature_ids,
        signed_response=signed,
        compatible_response=compatible,
        unmatched_signed_response=unmatched,
        direction=selected,
    )


__all__ = [
    "DirectionalResponseChannel",
    "ResponseDirection",
    "directional_response_channel",
]
