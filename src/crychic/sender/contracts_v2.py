"""Fold-frozen contracts for non-conserving detection and sender attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from crychic.core import stable_id

SENDER_ATTRIBUTION_V2_VERSION = "null_sender_entmax_attribution_v2"
_SCHEMA_VERSION = "2.0.0"
_PARENT_HEADS = {"parent_peak_raw", "parent_total_raw", "parent_mean_raw"}


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    result = tuple(sorted(_name(value, field_name=field_name) for value in values))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return result


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


class SenderV2CalibrationStatus(StrEnum):
    """Whether one fold-training calibration is usable on held-out rows."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, kw_only=True)
class SenderAttributionV2Spec:
    """Pre-registered empirical-null and sparse attribution policy."""

    parent_activity_head: str = "parent_mean_raw"
    minimum_calibration_subjects: int = 6
    mad_scale: float = 1.4826
    minimum_detection_scale: float = 1.0e-3
    attribution_temperature: float = 1.0
    maximum_abs_logit: float = 8.0
    ecdf_pseudocount: float = 0.5
    entmax_alpha: float = 1.5
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        parent_head = _name(
            self.parent_activity_head, field_name="parent_activity_head"
        )
        if parent_head not in _PARENT_HEADS:
            raise ValueError(
                f"parent_activity_head must be one of {sorted(_PARENT_HEADS)}"
            )
        minimum = self.minimum_calibration_subjects
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 4:
            raise ValueError("minimum_calibration_subjects must be an integer >= 4")
        mad_scale = _finite(self.mad_scale, field_name="mad_scale")
        scale_floor = _finite(
            self.minimum_detection_scale, field_name="minimum_detection_scale"
        )
        temperature = _finite(
            self.attribution_temperature, field_name="attribution_temperature"
        )
        maximum_logit = _finite(self.maximum_abs_logit, field_name="maximum_abs_logit")
        pseudocount = _finite(self.ecdf_pseudocount, field_name="ecdf_pseudocount")
        alpha = _finite(self.entmax_alpha, field_name="entmax_alpha")
        if mad_scale <= 0.0 or scale_floor <= 0.0 or temperature <= 0.0:
            raise ValueError("sender attribution scales must be positive")
        if maximum_logit <= 0.0:
            raise ValueError("maximum_abs_logit must be positive")
        if not 0.0 < pseudocount <= 1.0:
            raise ValueError("ecdf_pseudocount must lie in (0, 1]")
        if alpha != 1.5:
            raise ValueError("the v2 backend currently requires entmax_alpha=1.5")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "parent_activity_head", parent_head)
        object.__setattr__(self, "mad_scale", mad_scale)
        object.__setattr__(self, "minimum_detection_scale", scale_floor)
        object.__setattr__(self, "attribution_temperature", temperature)
        object.__setattr__(self, "maximum_abs_logit", maximum_logit)
        object.__setattr__(self, "ecdf_pseudocount", pseudocount)
        object.__setattr__(self, "entmax_alpha", alpha)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "sender_attribution_v2_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "attribution_temperature": self.attribution_temperature,
            "ecdf_pseudocount": self.ecdf_pseudocount,
            "entmax_alpha": self.entmax_alpha,
            "mad_scale": self.mad_scale,
            "maximum_abs_logit": self.maximum_abs_logit,
            "minimum_calibration_subjects": self.minimum_calibration_subjects,
            "minimum_detection_scale": self.minimum_detection_scale,
            "parent_activity_head": self.parent_activity_head,
            "schema_version": self.schema_version,
            "version": SENDER_ATTRIBUTION_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class ParentActivityCalibrationV2:
    """Subject-equal empirical reference values for one parent event."""

    receiver: str
    interaction_id: str
    n_subjects: int
    sorted_reference_values: tuple[float, ...]
    status: SenderV2CalibrationStatus | str
    reason_code: str | None
    calibration_id: str = field(init=False)

    def __post_init__(self) -> None:
        receiver = _name(self.receiver, field_name="receiver")
        interaction = _name(self.interaction_id, field_name="interaction_id")
        status = SenderV2CalibrationStatus(self.status)
        if (
            isinstance(self.n_subjects, bool)
            or not isinstance(self.n_subjects, int)
            or self.n_subjects < 0
        ):
            raise ValueError("n_subjects must be a non-negative integer")
        values = tuple(
            _finite(value, field_name="sorted_reference_values")
            for value in self.sorted_reference_values
        )
        if any(value < 0.0 for value in values) or values != tuple(sorted(values)):
            raise ValueError("parent reference values must be sorted and non-negative")
        if status is SenderV2CalibrationStatus.OBSERVED:
            if (
                not values
                or len(values) != self.n_subjects
                or self.reason_code is not None
            ):
                raise ValueError("observed parent calibration requires values")
        elif values or self.reason_code is None:
            raise ValueError("unavailable parent calibration requires a reason")
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "interaction_id", interaction)
        object.__setattr__(self, "sorted_reference_values", values)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "calibration_id",
            stable_id(
                "parent_activity_calibration_v2",
                self.to_dict(include_id=False),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "interaction_id": self.interaction_id,
            "n_subjects": self.n_subjects,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "sorted_reference_values": list(self.sorted_reference_values),
            "status": SenderV2CalibrationStatus(self.status).value,
        }
        return (
            {"calibration_id": self.calibration_id, **result} if include_id else result
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SenderDetectionCalibrationV2:
    """Robust training-fold location and scale for one sender event."""

    sender: str
    receiver: str
    interaction_id: str
    n_subjects: int
    center: float | None
    scale: float | None
    status: SenderV2CalibrationStatus | str
    reason_code: str | None
    calibration_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("sender", "receiver", "interaction_id"):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        status = SenderV2CalibrationStatus(self.status)
        if (
            isinstance(self.n_subjects, bool)
            or not isinstance(self.n_subjects, int)
            or self.n_subjects < 0
        ):
            raise ValueError("n_subjects must be a non-negative integer")
        center = (
            None if self.center is None else _finite(self.center, field_name="center")
        )
        scale = None if self.scale is None else _finite(self.scale, field_name="scale")
        if status is SenderV2CalibrationStatus.OBSERVED:
            if (
                center is None
                or center < 0.0
                or scale is None
                or scale <= 0.0
                or self.reason_code is not None
            ):
                raise ValueError("observed detection calibration requires center/scale")
        elif center is not None or scale is not None or self.reason_code is None:
            raise ValueError("unavailable detection calibration requires a reason")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "calibration_id",
            stable_id(
                "sender_detection_calibration_v2",
                self.to_dict(include_id=False),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "center": self.center,
            "interaction_id": self.interaction_id,
            "n_subjects": self.n_subjects,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "scale": self.scale,
            "sender": self.sender,
            "status": SenderV2CalibrationStatus(self.status).value,
        }
        return (
            {"calibration_id": self.calibration_id, **result} if include_id else result
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SenderAttributionFunctionalV2:
    """One outcome-agnostic attribution functional fitted on outer-fold subjects."""

    fold_id: str
    training_subject_ids: tuple[str, ...]
    training_input_digest: str
    activity_transform_id: str
    input_digest: str
    parent_calibrations: tuple[ParentActivityCalibrationV2, ...]
    detection_calibrations: tuple[SenderDetectionCalibrationV2, ...]
    spec: SenderAttributionV2Spec
    outcome_agnostic: bool = True
    formal_inference_allowed: bool = False
    functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "fold_id",
            "training_input_digest",
            "activity_transform_id",
            "input_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        subjects = _names(self.training_subject_ids, field_name="training_subject_ids")
        if len(self.input_digest) != 64:
            raise ValueError("input_digest must be a SHA-256-compatible digest")
        if not isinstance(self.spec, SenderAttributionV2Spec):
            raise TypeError("spec must be SenderAttributionV2Spec")
        supplied_parents = tuple(self.parent_calibrations)
        supplied_detections = tuple(self.detection_calibrations)
        if any(
            not isinstance(item, ParentActivityCalibrationV2)
            for item in supplied_parents
        ):
            raise TypeError("parent_calibrations contain invalid records")
        if any(
            not isinstance(item, SenderDetectionCalibrationV2)
            for item in supplied_detections
        ):
            raise TypeError("detection_calibrations contain invalid records")
        parents = tuple(
            sorted(
                supplied_parents,
                key=lambda item: (item.receiver, item.interaction_id),
            )
        )
        detections = tuple(
            sorted(
                supplied_detections,
                key=lambda item: (item.sender, item.receiver, item.interaction_id),
            )
        )
        parent_keys = [(item.receiver, item.interaction_id) for item in parents]
        detection_keys = [
            (item.sender, item.receiver, item.interaction_id) for item in detections
        ]
        if len(parent_keys) != len(set(parent_keys)) or len(detection_keys) != len(
            set(detection_keys)
        ):
            raise ValueError("sender v2 calibration keys must be unique")
        if not parents or not detections:
            raise ValueError("sender attribution functional requires calibrations")
        if self.outcome_agnostic is not True:
            raise ValueError("sender attribution v2 must remain outcome-agnostic")
        if self.formal_inference_allowed is not False:
            raise ValueError("sender attribution is descriptive, not formal inference")
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "parent_calibrations", parents)
        object.__setattr__(self, "detection_calibrations", detections)
        object.__setattr__(
            self,
            "functional_id",
            stable_id(
                "sender_attribution_functional_v2",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "activity_transform_id": self.activity_transform_id,
            "detection_calibrations": [
                item.to_dict() for item in self.detection_calibrations
            ],
            "fold_id": self.fold_id,
            "formal_inference_allowed": self.formal_inference_allowed,
            "input_digest": self.input_digest,
            "outcome_agnostic": self.outcome_agnostic,
            "parent_calibrations": [
                item.to_dict() for item in self.parent_calibrations
            ],
            "spec_id": self.spec.spec_id,
            "training_input_digest": self.training_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
            "version": SENDER_ATTRIBUTION_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_id": self.functional_id,
            **self._identity_payload(),
            "spec": self.spec.to_dict(),
        }


__all__ = [
    "SENDER_ATTRIBUTION_V2_VERSION",
    "ParentActivityCalibrationV2",
    "SenderAttributionFunctionalV2",
    "SenderAttributionV2Spec",
    "SenderDetectionCalibrationV2",
    "SenderV2CalibrationStatus",
]
