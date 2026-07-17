"""Selected-penalty inner-OOF calibration for subject-family gains.

The producer consumes only outer-training inner-fold applications.  It records
the exact loss arrays needed to reproduce the released subject-family bounded
gain and a receiver-local positive-gain percentile mapping.  The mapping is a
descriptive calibration aid; it does not provide formal inference.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from crychic.core import ContractError, stable_id

if TYPE_CHECKING:
    from crychic.scoring.downstream import (
        IncrementalDownstreamApplication,
        IncrementalDownstreamFunctional,
    )

    from .tuning import PenaltyTuningArtifact

_SPEC_SCHEMA_VERSION = "1.0.0"
_ARTIFACT_SCHEMA_VERSION = "1.0.0"
_ARTIFACT_PRODUCER = "crychic.attribution.selected_penalty_inner_oof_gain.v1"
_OBSERVED_STATUS = "observed"
_NOT_ESTIMABLE_STATUS = "not_estimable"
_CALIBRATION_STATUS = (
    "outer_training_selected_penalty_inner_oof_gain_calibration_descriptive_v1"
)
_GAIN_SEMANTICS = "subject_bounded_incremental_gain_v1"
GAIN_CALIBRATION_PERCENTILE_POLICY = (
    "positive_subject_family_observation_equal_right_ecdf_"
    "piecewise_linear_zero_preserving_v1"
)
_PERCENTILE_POLICY = GAIN_CALIBRATION_PERCENTILE_POLICY
_SUPPORTED_LOSS_DESIGNS = frozenset(
    {
        "fully_paired_subject_contrasts_v1",
        "independent_subject_pseudocontrasts_v1",
        "mixed_subject_equal_full_prediction_loss_v1",
    }
)
_VALIDATION_ESTIMAND_BY_LOSS_DESIGN = {
    "fully_paired_subject_contrasts_v1": ("paired_subject_contrast_prediction_loss_v1"),
    "independent_subject_pseudocontrasts_v1": (
        "independent_subject_full_prediction_loss_v1"
    ),
    "mixed_subject_equal_full_prediction_loss_v1": (
        "mixed_subject_equal_full_prediction_loss_v1"
    ),
}

# Imported by the workflow module together with the private finalizer.
_WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN = object()


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _canonical_names(
    values: Sequence[str], *, field_name: str, minimum: int = 1
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an identifier sequence")
    normalized = tuple(
        _required_name(value, field_name=field_name) for value in tuple(values)
    )
    if len(normalized) < minimum or len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{field_name} must contain at least {minimum} unique identifiers"
        )
    return normalized


def _minimum_integer(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _canonical_float_array(values: object, *, field_name: str) -> np.ndarray:
    result = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(np.isinf(result)):
        raise ValueError(f"{field_name} must not contain infinite values")
    result[np.isnan(result)] = np.nan
    result[result == 0.0] = 0.0
    return cast(np.ndarray, result)


def _immutable_float_array(values: object, *, field_name: str) -> np.ndarray:
    canonical = _canonical_float_array(values, field_name=field_name)
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _immutable_bool_array(values: object, *, field_name: str) -> np.ndarray:
    canonical = np.asarray(values, dtype="|b1", order="C").copy(order="C")
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="|b1").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _is_immutable_byte_backed(values: np.ndarray) -> bool:
    if values.flags.writeable or not values.flags.c_contiguous:
        return False
    base: object = values
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _array_digest(values: object, *, dtype: str, field_name: str) -> str:
    if dtype == "<f8":
        canonical = _canonical_float_array(values, field_name=field_name)
    elif dtype == "|b1":
        canonical = np.asarray(values, dtype="|b1", order="C").copy(order="C")
    else:  # pragma: no cover - private caller invariant
        raise ValueError("unsupported calibration digest dtype")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(dtype.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class GainCalibrationSpec:
    """Pre-registered minimum support for receiver gain calibration."""

    min_inner_folds: int = 2
    min_subjects: int = 8
    min_supported_families: int = 3
    min_subjects_per_family: int = 4
    min_positive_observations: int = 12
    min_distinct_positive_gains: int = 5
    schema_version: str = _SPEC_SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            "min_inner_folds": _minimum_integer(
                self.min_inner_folds, field_name="min_inner_folds", minimum=2
            ),
            "min_subjects": _minimum_integer(
                self.min_subjects, field_name="min_subjects", minimum=2
            ),
            "min_supported_families": _minimum_integer(
                self.min_supported_families,
                field_name="min_supported_families",
                minimum=1,
            ),
            "min_subjects_per_family": _minimum_integer(
                self.min_subjects_per_family,
                field_name="min_subjects_per_family",
                minimum=1,
            ),
            "min_positive_observations": _minimum_integer(
                self.min_positive_observations,
                field_name="min_positive_observations",
                minimum=1,
            ),
            "min_distinct_positive_gains": _minimum_integer(
                self.min_distinct_positive_gains,
                field_name="min_distinct_positive_gains",
                minimum=2,
            ),
        }
        if self.schema_version != _SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"gain calibration schema_version must be {_SPEC_SCHEMA_VERSION}"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "gain_calibration_spec",
                {
                    **values,
                    "percentile_policy": _PERCENTILE_POLICY,
                    "schema_version": self.schema_version,
                },
                schema_version="1",
            ),
        )

    def _require_intact(self) -> None:
        try:
            repeated = GainCalibrationSpec(
                min_inner_folds=self.min_inner_folds,
                min_subjects=self.min_subjects,
                min_supported_families=self.min_supported_families,
                min_subjects_per_family=self.min_subjects_per_family,
                min_positive_observations=self.min_positive_observations,
                min_distinct_positive_gains=self.min_distinct_positive_gains,
                schema_version=self.schema_version,
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "Gain calibration specification failed integrity validation",
                code="gain_calibration_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the pre-registered calibration specification",
            ) from error
        if repeated.spec_id != self.spec_id:
            raise ContractError(
                "Gain calibration specification failed integrity validation",
                code="gain_calibration_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the pre-registered calibration specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "schema_version": self.schema_version,
            "min_inner_folds": self.min_inner_folds,
            "min_subjects": self.min_subjects,
            "min_supported_families": self.min_supported_families,
            "min_subjects_per_family": self.min_subjects_per_family,
            "min_positive_observations": self.min_positive_observations,
            "min_distinct_positive_gains": self.min_distinct_positive_gains,
            "percentile_policy": _PERCENTILE_POLICY,
        }


@dataclass(frozen=True, slots=True, init=False)
class SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact:
    """Train-only selected-penalty subject-family gain calibration artifact."""

    spec: GainCalibrationSpec
    receiver: str
    contrast_name: str
    outer_fold_id: str
    training_subject_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    validation_inner_fold_ids: tuple[str, ...]
    selected_evaluation_ids: tuple[str, ...]
    inner_fold_lineage: tuple[tuple[str, str, str, str, str], ...]
    loss_design: str
    loss_aggregation: str
    penalty_validation_loss_estimand: str
    family_gain_estimand: str
    null_loss_floor: float
    tuning_spec_id: str
    tuning_id: str
    tuning_scope_id: str
    inner_fold_plan_id: str
    selected_candidate_id: str
    outer_incremental_functional_id: str
    outer_selected_resolved_penalty_id: str
    subject_null_losses: np.ndarray
    subject_family_losses: np.ndarray
    bounded_subject_family_gains: np.ndarray
    estimable_mask: np.ndarray
    structural_zero_mask: np.ndarray
    positive_gain_source_knots: np.ndarray
    positive_gain_percentile_knots: np.ndarray
    n_supported_families: int
    n_positive_observations: int
    n_distinct_positive_gains: int
    status: str
    reason_code: str | None
    certification_status: str
    artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact is producer-owned; "
            "use the receiver incremental workflow"
        )

    @property
    def is_estimable(self) -> bool:
        return self.status == _OBSERVED_STATUS

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    @property
    def percentile_policy(self) -> str:
        return _PERCENTILE_POLICY

    @property
    def positive_gain_source_knots_digest(self) -> str:
        return _array_digest(
            self.positive_gain_source_knots,
            dtype="<f8",
            field_name="positive_gain_source_knots",
        )

    @property
    def positive_gain_percentile_knots_digest(self) -> str:
        return _array_digest(
            self.positive_gain_percentile_knots,
            dtype="<f8",
            field_name="positive_gain_percentile_knots",
        )

    def _array_digests(self) -> dict[str, str]:
        return {
            "bounded_subject_family_gains_digest": _array_digest(
                self.bounded_subject_family_gains,
                dtype="<f8",
                field_name="bounded_subject_family_gains",
            ),
            "estimable_mask_digest": _array_digest(
                self.estimable_mask,
                dtype="|b1",
                field_name="estimable_mask",
            ),
            "positive_gain_percentile_knots_digest": (
                self.positive_gain_percentile_knots_digest
            ),
            "positive_gain_source_knots_digest": (
                self.positive_gain_source_knots_digest
            ),
            "structural_zero_mask_digest": _array_digest(
                self.structural_zero_mask,
                dtype="|b1",
                field_name="structural_zero_mask",
            ),
            "subject_family_losses_digest": _array_digest(
                self.subject_family_losses,
                dtype="<f8",
                field_name="subject_family_losses",
            ),
            "subject_null_losses_digest": _array_digest(
                self.subject_null_losses,
                dtype="<f8",
                field_name="subject_null_losses",
            ),
        }

    def _identity_payload(self) -> dict[str, object]:
        return {
            **self._array_digests(),
            "calibration_semantics": _GAIN_SEMANTICS,
            "certification_status": self.certification_status,
            "contrast_name": self.contrast_name,
            "family_gain_estimand": self.family_gain_estimand,
            "family_ids": list(self.family_ids),
            "feature_ids": list(self.feature_ids),
            "inner_fold_lineage": [list(item) for item in self.inner_fold_lineage],
            "inner_fold_plan_id": self.inner_fold_plan_id,
            "loss_aggregation": self.loss_aggregation,
            "loss_design": self.loss_design,
            "penalty_validation_loss_estimand": (self.penalty_validation_loss_estimand),
            "n_distinct_positive_gains": self.n_distinct_positive_gains,
            "n_positive_observations": self.n_positive_observations,
            "n_supported_families": self.n_supported_families,
            "null_loss_floor": self.null_loss_floor,
            "outer_fold_id": self.outer_fold_id,
            "outer_incremental_functional_id": (self.outer_incremental_functional_id),
            "outer_selected_resolved_penalty_id": (
                self.outer_selected_resolved_penalty_id
            ),
            "percentile_policy": _PERCENTILE_POLICY,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_evaluation_ids": list(self.selected_evaluation_ids),
            "spec_id": self.spec.spec_id,
            "status": self.status,
            "training_subject_ids": list(self.training_subject_ids),
            "tuning_id": self.tuning_id,
            "tuning_scope_id": self.tuning_scope_id,
            "tuning_spec_id": self.tuning_spec_id,
            "validation_inner_fold_ids": list(self.validation_inner_fold_ids),
        }

    def _require_intact(self) -> None:
        try:
            self.spec._require_intact()
            n_subjects = len(self.training_subject_ids)
            n_families = len(self.family_ids)
            expected_shape = (n_subjects, n_families)
            arrays = (
                self.subject_null_losses,
                self.subject_family_losses,
                self.bounded_subject_family_gains,
                self.estimable_mask,
                self.structural_zero_mask,
                self.positive_gain_source_knots,
                self.positive_gain_percentile_knots,
            )
            if not all(_is_immutable_byte_backed(array) for array in arrays):
                raise ValueError("calibration arrays must be immutable and byte-backed")
            if self.subject_null_losses.shape != (n_subjects,):
                raise ValueError("subject_null_losses shape is invalid")
            if any(
                array.shape != expected_shape
                for array in (
                    self.subject_family_losses,
                    self.bounded_subject_family_gains,
                    self.estimable_mask,
                    self.structural_zero_mask,
                )
            ):
                raise ValueError("subject-family calibration array shape is invalid")
            recomputed = _bounded_gains_from_losses(
                self.subject_null_losses,
                self.subject_family_losses,
                null_loss_floor=self.null_loss_floor,
            )
            gains, estimable, structural = recomputed
            if not _equal_nan(gains, self.bounded_subject_family_gains):
                raise ValueError("bounded gains do not match stored losses")
            if not np.array_equal(estimable, self.estimable_mask):
                raise ValueError("estimable mask does not match stored losses")
            if not np.array_equal(structural, self.structural_zero_mask):
                raise ValueError("structural-zero mask does not match stored losses")
            support = _calibration_support(
                self.spec,
                bounded_gains=gains,
                estimable_mask=estimable,
                n_inner_folds=len(self.selected_evaluation_ids),
                n_subjects=n_subjects,
            )
            (
                expected_status,
                expected_reason,
                supported_families,
                positive,
                distinct,
            ) = support
            source_knots, percentile_knots = _percentile_knots(
                positive,
                estimable=expected_status == _OBSERVED_STATUS,
            )
            valid = (
                self._producer_marker == _ARTIFACT_PRODUCER
                and self.certification_status == _CALIBRATION_STATUS
                and self.loss_design in _SUPPORTED_LOSS_DESIGNS
                and self.penalty_validation_loss_estimand
                == _VALIDATION_ESTIMAND_BY_LOSS_DESIGN[self.loss_design]
                and self.status == expected_status
                and self.reason_code == expected_reason
                and self.n_supported_families == len(supported_families)
                and self.n_positive_observations == len(positive)
                and self.n_distinct_positive_gains == len(distinct)
                and _equal_nan(source_knots, self.positive_gain_source_knots)
                and _equal_nan(percentile_knots, self.positive_gain_percentile_knots)
                and len(self.validation_inner_fold_ids) == n_subjects
                and len(self.selected_evaluation_ids) == len(self.inner_fold_lineage)
                and stable_id(
                    "selected_penalty_inner_oof_family_gain_calibration",
                    self._identity_payload(),
                    schema_version="1",
                )
                == self.artifact_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Selected-penalty gain calibration failed integrity validation",
                code="gain_calibration_artifact_integrity_violation",
                field="artifact_id",
                remediation=(
                    "Rebuild calibration from intact selected inner applications"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Selected-penalty gain calibration failed integrity validation",
                code="gain_calibration_artifact_integrity_violation",
                field="artifact_id",
                remediation=(
                    "Rebuild calibration from intact selected inner applications"
                ),
            )

    def calibrate_gain(self, value: object) -> float | None:
        """Map one unit-interval gain to its train-only positive-gain percentile."""

        self._require_intact()
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("gain must be numeric, not boolean")
        numeric = float(cast(Any, value))
        if math.isnan(numeric):
            return None
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError("gain must lie in [0, 1] or be missing")
        if numeric == 0.0:
            return 0.0
        if not self.is_estimable:
            return None
        return float(
            np.interp(
                numeric,
                self.positive_gain_source_knots,
                self.positive_gain_percentile_knots,
                left=0.0,
                right=1.0,
            )
        )

    def calibrate_gains(self, values: object) -> np.ndarray:
        """Vectorized gain calibration preserving zero and missing values."""

        self._require_intact()
        source = np.asarray(values, dtype=np.float64)
        if np.any(np.isinf(source)) or np.any(
            np.isfinite(source) & ((source < 0.0) | (source > 1.0))
        ):
            raise ValueError("gains must lie in [0, 1] or be missing")
        result = np.full(source.shape, np.nan, dtype=np.float64)
        zero = np.isfinite(source) & (source == 0.0)
        result[zero] = 0.0
        positive = np.isfinite(source) & (source > 0.0)
        if self.is_estimable:
            result[positive] = np.interp(
                source[positive],
                self.positive_gain_source_knots,
                self.positive_gain_percentile_knots,
                left=0.0,
                right=1.0,
            )
        return cast(np.ndarray, result)

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "artifact_id": self.artifact_id,
            **self._identity_payload(),
            "formal_inference_allowed": False,
            "is_estimable": self.is_estimable,
            "spec": self.spec.to_dict(),
        }


def _equal_nan(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return left_array.shape == right_array.shape and bool(
        np.array_equal(left_array, right_array, equal_nan=True)
    )


def _bounded_gains_from_losses(
    subject_null_losses: object,
    subject_family_losses: object,
    *,
    null_loss_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    null_losses = _canonical_float_array(
        subject_null_losses, field_name="subject_null_losses"
    )
    family_losses = _canonical_float_array(
        subject_family_losses, field_name="subject_family_losses"
    )
    if null_losses.ndim != 1 or family_losses.ndim != 2:
        raise ValueError("gain calibration losses must be a vector and matrix")
    if family_losses.shape[0] != len(null_losses):
        raise ValueError("subject loss arrays do not align")
    if not math.isfinite(null_loss_floor) or null_loss_floor <= 0:
        raise ValueError("null_loss_floor must be finite and positive")
    if np.any(np.isfinite(null_losses) & (null_losses < 0)) or np.any(
        np.isfinite(family_losses) & (family_losses < 0)
    ):
        raise ValueError("prediction losses must be non-negative")
    estimable = np.isfinite(null_losses)[:, np.newaxis] & np.isfinite(family_losses)
    gains = np.full(family_losses.shape, np.nan, dtype=np.float64)
    structural = np.zeros(family_losses.shape, dtype=bool)
    positive_denominator = estimable & (null_losses[:, np.newaxis] > null_loss_floor)
    zero_denominator = estimable & ~positive_denominator
    gains[zero_denominator] = 0.0
    structural[zero_denominator] = True
    raw = np.zeros(family_losses.shape, dtype=np.float64)
    np.divide(
        null_losses[:, np.newaxis] - family_losses,
        null_losses[:, np.newaxis],
        out=raw,
        where=positive_denominator,
    )
    negligible = positive_denominator & (raw <= null_loss_floor)
    gains[negligible] = 0.0
    structural[negligible] = True
    positive = positive_denominator & ~negligible
    gains[positive] = np.clip(raw[positive], 0.0, 1.0)
    gains[gains == 0.0] = 0.0
    return gains, estimable, structural


def _calibration_support(
    spec: GainCalibrationSpec,
    *,
    bounded_gains: np.ndarray,
    estimable_mask: np.ndarray,
    n_inner_folds: int,
    n_subjects: int,
) -> tuple[str, str | None, np.ndarray, np.ndarray, np.ndarray]:
    finite_counts = np.sum(estimable_mask, axis=0)
    supported_families = np.flatnonzero(finite_counts >= spec.min_subjects_per_family)
    selected = (
        np.empty(0, dtype=np.float64)
        if not len(supported_families)
        else bounded_gains[:, supported_families]
    )
    positive = np.asarray(
        selected[np.isfinite(selected) & (selected > 0.0)], dtype=np.float64
    )
    distinct = np.unique(positive)
    reason: str | None = None
    if n_inner_folds < spec.min_inner_folds:
        reason = "insufficient_inner_oof_folds"
    elif n_subjects < spec.min_subjects:
        reason = "insufficient_inner_oof_subjects"
    elif len(supported_families) < spec.min_supported_families:
        reason = "insufficient_supported_gain_families"
    elif len(positive) < spec.min_positive_observations:
        reason = "insufficient_positive_gain_observations"
    elif len(distinct) < spec.min_distinct_positive_gains:
        reason = "insufficient_distinct_positive_gains"
    status = _OBSERVED_STATUS if reason is None else _NOT_ESTIMABLE_STATUS
    return status, reason, supported_families, positive, distinct


def _percentile_knots(
    positive_gains: np.ndarray, *, estimable: bool
) -> tuple[np.ndarray, np.ndarray]:
    if not estimable:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    positive = np.sort(np.asarray(positive_gains, dtype=np.float64))
    if not len(positive) or np.any(~np.isfinite(positive)) or np.any(positive <= 0):
        raise ValueError("observed percentile calibration requires positive gains")
    unique, counts = np.unique(positive, return_counts=True)
    source = np.concatenate((np.asarray([0.0]), unique))
    percentile = np.concatenate(
        (np.asarray([0.0]), np.cumsum(counts, dtype=np.float64) / len(positive))
    )
    return source, percentile


def _calibration_parent_error(message: str, *, field: str) -> ContractError:
    return ContractError(
        message,
        code="gain_calibration_parent_mismatch",
        field=field,
        remediation="Use the exact selected-candidate inner fit/apply lineage",
    )


def _finalize_selected_penalty_inner_oof_gain_calibration(
    *,
    _producer_token: object,
    spec: GainCalibrationSpec,
    tuning_artifact: PenaltyTuningArtifact,
    inner_functionals: Sequence[IncrementalDownstreamFunctional],
    inner_applications: Sequence[IncrementalDownstreamApplication],
    outer_final_functional: IncrementalDownstreamFunctional,
) -> SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact:
    """Finalize selected-candidate gain calibration from outer-training OOF rows."""

    if _producer_token is not _WORKFLOW_GAIN_CALIBRATION_PRODUCER_TOKEN:
        raise TypeError("selected-penalty gain calibration is workflow-producer-owned")
    if not isinstance(spec, GainCalibrationSpec):
        raise TypeError("spec must be GainCalibrationSpec")
    spec._require_intact()

    # Local imports avoid an attribution/scoring package initialization cycle.
    from crychic.scoring.downstream import (
        IncrementalDownstreamApplication,
        IncrementalDownstreamFunctional,
    )

    from .tuning import PenaltyTuningArtifact

    if not isinstance(tuning_artifact, PenaltyTuningArtifact):
        raise TypeError("tuning_artifact must be PenaltyTuningArtifact")
    tuning_artifact._require_intact()
    if not tuning_artifact.is_oof_certified:
        raise _calibration_parent_error(
            "Gain calibration requires verified subject-blocked tuning",
            field="tuning_artifact",
        )
    selected_candidate_id = cast(str, tuning_artifact.selected_candidate_id)
    if not isinstance(outer_final_functional, IncrementalDownstreamFunctional):
        raise TypeError(
            "outer_final_functional must be IncrementalDownstreamFunctional"
        )
    outer_final_functional._require_intact()
    if (
        outer_final_functional.penalty_candidate_id != selected_candidate_id
        or outer_final_functional.resolved_penalty_id is None
        or outer_final_functional.training_subject_ids
        != tuning_artifact.training_subject_ids
    ):
        raise _calibration_parent_error(
            "Outer functional does not use the selected penalty and tuning scope",
            field="outer_final_functional",
        )
    if tuning_artifact.inner_fold_plan_id is None:
        raise _calibration_parent_error(
            "Gain calibration requires an inner fold plan",
            field="inner_fold_plan_id",
        )
    validation_loss_estimand = tuning_artifact.validation_loss_estimand
    if (
        validation_loss_estimand is None
        or validation_loss_estimand.value
        != _VALIDATION_ESTIMAND_BY_LOSS_DESIGN[outer_final_functional.loss_design]
    ):
        raise _calibration_parent_error(
            "Tuning and final functional use incompatible loss estimands",
            field="validation_loss_estimand",
        )

    functionals = tuple(inner_functionals)
    applications = tuple(inner_applications)
    if not functionals or any(
        not isinstance(item, IncrementalDownstreamFunctional) for item in functionals
    ):
        raise TypeError(
            "inner_functionals must contain IncrementalDownstreamFunctional values"
        )
    if not applications or any(
        not isinstance(item, IncrementalDownstreamApplication) for item in applications
    ):
        raise TypeError(
            "inner_applications must contain IncrementalDownstreamApplication values"
        )
    for functional in functionals:
        functional._require_intact()
    for application in applications:
        application._require_intact()

    selected_evaluations = tuple(
        sorted(
            (
                item
                for item in tuning_artifact.evaluations
                if item.candidate_id == selected_candidate_id
            ),
            key=lambda item: item.inner_fold_id,
        )
    )
    expected_fold_ids = tuple(sorted(tuning_artifact.inner_fold_ids))
    if tuple(
        item.inner_fold_id for item in selected_evaluations
    ) != expected_fold_ids or any(
        item.status != _OBSERVED_STATUS for item in selected_evaluations
    ):
        raise _calibration_parent_error(
            "Selected candidate lacks complete observed inner-fold evaluations",
            field="selected_evaluation_ids",
        )
    functionals_by_id = {item.incremental_functional_id: item for item in functionals}
    applications_by_id = {item.application_id: item for item in applications}
    if len(functionals_by_id) != len(functionals) or len(applications_by_id) != len(
        applications
    ):
        raise _calibration_parent_error(
            "Selected inner parent IDs must be unique", field="inner_functionals"
        )
    expected_functional_ids = {
        cast(str, item.training_functional_id) for item in selected_evaluations
    }
    expected_application_ids = {
        cast(str, item.heldout_application_id) for item in selected_evaluations
    }
    if (
        set(functionals_by_id) != expected_functional_ids
        or set(applications_by_id) != expected_application_ids
    ):
        raise _calibration_parent_error(
            "Selected inner parents do not exactly match tuning evaluations",
            field="inner_applications",
        )

    subject_rows: dict[str, tuple[str, float, np.ndarray]] = {}
    fold_lineage: list[tuple[str, str, str, str, str]] = []
    loss_aggregation: str | None = None
    for evaluation in selected_evaluations:
        functional = functionals_by_id[cast(str, evaluation.training_functional_id)]
        application = applications_by_id[cast(str, evaluation.heldout_application_id)]
        if (
            functional.penalty_candidate_id != selected_candidate_id
            or functional.fold_id != evaluation.inner_fold_id
            or functional.training_subject_ids != evaluation.inner_training_subject_ids
            or functional.receiver != outer_final_functional.receiver
            or functional.contrast_name != outer_final_functional.contrast_name
            or functional.family_ids != outer_final_functional.family_ids
            or functional.feature_ids != outer_final_functional.feature_ids
            or functional.loss_design != outer_final_functional.loss_design
            or functional.family_gain_estimand
            != outer_final_functional.family_gain_estimand
            or functional.null_loss_floor != outer_final_functional.null_loss_floor
            or functional.penalty_scale_resolution_id != evaluation.scale_resolution_id
            or functional.resolved_penalty_id != evaluation.resolved_penalty_id
            or application.incremental_functional_id
            != functional.incremental_functional_id
            or application.application_id != evaluation.heldout_application_id
            or application.status != _OBSERVED_STATUS
            or application.subject_ids != evaluation.validation_subject_ids
            or application.family_ids != outer_final_functional.family_ids
        ):
            raise _calibration_parent_error(
                "Selected inner fit/apply lineage is incompatible",
                field="inner_applications",
            )
        if loss_aggregation is None:
            loss_aggregation = application.loss_aggregation
        elif loss_aggregation != application.loss_aggregation:
            raise _calibration_parent_error(
                "Selected inner applications mix loss aggregation policies",
                field="loss_aggregation",
            )
        if application.subject_null_losses.shape != (len(application.subject_ids),):
            raise _calibration_parent_error(
                "Selected inner null losses do not align subjects",
                field="subject_null_losses",
            )
        expected_family_shape = (
            len(application.subject_ids),
            len(outer_final_functional.family_ids),
        )
        if application.subject_family_losses.shape != expected_family_shape:
            raise _calibration_parent_error(
                "Selected inner family losses do not align subjects and families",
                field="subject_family_losses",
            )
        for index, subject_id in enumerate(application.subject_ids):
            if subject_id in subject_rows:
                raise ContractError(
                    "Inner validation subjects are not exact OOF",
                    code="gain_calibration_oof_coverage_mismatch",
                    field="training_subject_ids",
                    remediation="Validate every outer-training subject exactly once",
                )
            subject_rows[subject_id] = (
                evaluation.inner_fold_id,
                float(application.subject_null_losses[index]),
                np.asarray(application.subject_family_losses[index], dtype=np.float64),
            )
        fold_lineage.append(
            (
                evaluation.inner_fold_id,
                functional.incremental_functional_id,
                application.application_id,
                cast(str, functional.penalty_scale_resolution_id),
                cast(str, functional.resolved_penalty_id),
            )
        )

    training_subject_ids = tuple(tuning_artifact.training_subject_ids)
    if tuple(sorted(subject_rows)) != training_subject_ids:
        raise ContractError(
            "Inner validation subjects do not exactly cover outer training",
            code="gain_calibration_oof_coverage_mismatch",
            field="training_subject_ids",
            remediation="Validate every outer-training subject exactly once",
        )
    validation_fold_ids = tuple(subject_rows[item][0] for item in training_subject_ids)
    subject_null_losses = np.asarray(
        [subject_rows[item][1] for item in training_subject_ids], dtype=np.float64
    )
    subject_family_losses = np.vstack(
        [subject_rows[item][2] for item in training_subject_ids]
    )
    bounded_gains, estimable_mask, structural_zero_mask = _bounded_gains_from_losses(
        subject_null_losses,
        subject_family_losses,
        null_loss_floor=outer_final_functional.null_loss_floor,
    )
    status, reason, supported, positive, distinct = _calibration_support(
        spec,
        bounded_gains=bounded_gains,
        estimable_mask=estimable_mask,
        n_inner_folds=len(selected_evaluations),
        n_subjects=len(training_subject_ids),
    )
    source_knots, percentile_knots = _percentile_knots(
        positive, estimable=status == _OBSERVED_STATUS
    )

    self = object.__new__(SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact)
    values: dict[str, object] = {
        "spec": spec,
        "receiver": outer_final_functional.receiver,
        "contrast_name": outer_final_functional.contrast_name,
        "outer_fold_id": outer_final_functional.fold_id,
        "training_subject_ids": training_subject_ids,
        "family_ids": tuple(outer_final_functional.family_ids),
        "feature_ids": tuple(outer_final_functional.feature_ids),
        "validation_inner_fold_ids": validation_fold_ids,
        "selected_evaluation_ids": tuple(
            item.evaluation_id for item in selected_evaluations
        ),
        "inner_fold_lineage": tuple(fold_lineage),
        "loss_design": outer_final_functional.loss_design,
        "loss_aggregation": cast(str, loss_aggregation),
        "penalty_validation_loss_estimand": validation_loss_estimand.value,
        "family_gain_estimand": outer_final_functional.family_gain_estimand,
        "null_loss_floor": outer_final_functional.null_loss_floor,
        "tuning_spec_id": tuning_artifact.spec.spec_id,
        "tuning_id": tuning_artifact.tuning_id,
        "tuning_scope_id": tuning_artifact.tuning_scope_id,
        "inner_fold_plan_id": tuning_artifact.inner_fold_plan_id,
        "selected_candidate_id": selected_candidate_id,
        "outer_incremental_functional_id": (
            outer_final_functional.incremental_functional_id
        ),
        "outer_selected_resolved_penalty_id": (
            outer_final_functional.resolved_penalty_id
        ),
        "subject_null_losses": _immutable_float_array(
            subject_null_losses, field_name="subject_null_losses"
        ),
        "subject_family_losses": _immutable_float_array(
            subject_family_losses, field_name="subject_family_losses"
        ),
        "bounded_subject_family_gains": _immutable_float_array(
            bounded_gains, field_name="bounded_subject_family_gains"
        ),
        "estimable_mask": _immutable_bool_array(
            estimable_mask, field_name="estimable_mask"
        ),
        "structural_zero_mask": _immutable_bool_array(
            structural_zero_mask, field_name="structural_zero_mask"
        ),
        "positive_gain_source_knots": _immutable_float_array(
            source_knots, field_name="positive_gain_source_knots"
        ),
        "positive_gain_percentile_knots": _immutable_float_array(
            percentile_knots, field_name="positive_gain_percentile_knots"
        ),
        "n_supported_families": len(supported),
        "n_positive_observations": len(positive),
        "n_distinct_positive_gains": len(distinct),
        "status": status,
        "reason_code": reason,
        "certification_status": _CALIBRATION_STATUS,
        "_producer_marker": _ARTIFACT_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "artifact_id",
        stable_id(
            "selected_penalty_inner_oof_family_gain_calibration",
            self._identity_payload(),
            schema_version="1",
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "GAIN_CALIBRATION_PERCENTILE_POLICY",
    "GainCalibrationSpec",
    "SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact",
]
