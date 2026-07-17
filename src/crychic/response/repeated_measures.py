"""Subject-cluster receiver effects for repeated-measures designs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import FrozenRepeatedMeasuresDesign

_METHOD = "formula_ols_subject_cluster_cr1_v1"
_UNCERTAINTY = "subject_cluster_cr1_diagnostic_only_no_calibrated_p_or_q"
_SCHEMA_VERSION = "1.0.0"
_PRODUCER_MARKER = "crychic.response.repeated_measures_receiver_effect.v1"


def _names(
    values: Sequence[str], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if (not result and not allow_empty) or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{field_name} must contain canonical non-empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique names")
    return result


def _numeric_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(np.isinf(array)):
        raise ValueError("receiver response values cannot contain infinity")
    array[np.isnan(array)] = np.nan
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _identity_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _vector_estimable(
    matrix: np.ndarray, vector: np.ndarray, *, tolerance: float = 1.0e-8
) -> bool:
    if matrix.size == 0 or np.linalg.norm(vector) <= tolerance:
        return False
    projection = np.linalg.pinv(matrix) @ matrix
    residual = vector - vector @ projection
    return bool(
        np.linalg.norm(residual)
        <= tolerance * max(1.0, float(np.linalg.norm(vector)))
    )


@dataclass(frozen=True, slots=True)
class RepeatedMeasuresFeatureEffect:
    """One exploratory receiver feature effect and cluster diagnostics."""

    feature_id: str
    effect: float
    diagnostic_standard_error: float
    raw_precision: float
    n_input_samples: int
    n_model_cells: int
    n_subject_clusters: int
    n_contrast_subject_clusters: int
    n_context_subjects_min: int
    n_repeated_subject_clusters: int
    n_complete_contrast_subjects: int
    design_rank: int
    residual_df: int | None
    cluster_df: int | None
    condition_number: float
    status: str
    reason_code: str | None
    diagnostic_codes: tuple[str, ...]
    effect_id: str

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    def _identity_payload(self, *, response_input_id: str) -> dict[str, object]:
        return {
            "cluster_df": self.cluster_df,
            "condition_number": _identity_float(self.condition_number),
            "design_rank": self.design_rank,
            "diagnostic_codes": list(self.diagnostic_codes),
            "diagnostic_standard_error": _identity_float(
                self.diagnostic_standard_error
            ),
            "effect": _identity_float(self.effect),
            "feature_id": self.feature_id,
            "n_complete_contrast_subjects": self.n_complete_contrast_subjects,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_input_samples": self.n_input_samples,
            "n_model_cells": self.n_model_cells,
            "n_context_subjects_min": self.n_context_subjects_min,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "n_subject_clusters": self.n_subject_clusters,
            "raw_precision": _identity_float(self.raw_precision),
            "reason_code": self.reason_code,
            "residual_df": self.residual_df,
            "response_input_id": response_input_id,
            "status": self.status,
        }

    def as_record(self) -> dict[str, object]:
        """Return a flat descriptive record without inferential fields."""

        return {
            "effect_id": self.effect_id,
            "feature_id": self.feature_id,
            "effect": self.effect,
            "effect_semantics": "declared_context_contrast_adjusted_mean",
            "diagnostic_standard_error": self.diagnostic_standard_error,
            "raw_precision": self.raw_precision,
            "n_input_samples": self.n_input_samples,
            "n_model_cells": self.n_model_cells,
            "n_subject_clusters": self.n_subject_clusters,
            "n_contrast_subject_clusters": self.n_contrast_subject_clusters,
            "n_context_subjects_min": self.n_context_subjects_min,
            "n_repeated_subject_clusters": self.n_repeated_subject_clusters,
            "n_complete_contrast_subjects": self.n_complete_contrast_subjects,
            "design_rank": self.design_rank,
            "residual_df": self.residual_df,
            "cluster_df": self.cluster_df,
            "condition_number": self.condition_number,
            "status": self.status,
            "reason_code": self.reason_code,
            "diagnostic_codes": self.diagnostic_codes,
            "method": _METHOD,
            "uncertainty_semantics": _UNCERTAINTY,
            "formal_inference_allowed": False,
        }


@dataclass(frozen=True, slots=True, init=False)
class RepeatedMeasuresReceiverEffect:
    """Producer-owned multi-feature repeated-measures receiver result."""

    design_id: str
    contrast_name: str
    receiver: str
    sample_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    sample_values_digest: str
    response_input_id: str
    feature_effects: tuple[RepeatedMeasuresFeatureEffect, ...]
    artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "RepeatedMeasuresReceiverEffect is producer-owned; "
            "use fit_repeated_measures_receiver_effect()"
        )

    @classmethod
    def _from_fit(
        cls,
        *,
        design: FrozenRepeatedMeasuresDesign,
        receiver: str,
        sample_ids: tuple[str, ...],
        feature_ids: tuple[str, ...],
        sample_values_digest: str,
        response_input_id: str,
        feature_effects: tuple[RepeatedMeasuresFeatureEffect, ...],
    ) -> RepeatedMeasuresReceiverEffect:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "design_id": design.design_id,
            "contrast_name": design.contrast.name,
            "receiver": receiver,
            "sample_ids": sample_ids,
            "feature_ids": feature_ids,
            "sample_values_digest": sample_values_digest,
            "response_input_id": response_input_id,
            "feature_effects": feature_effects,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "artifact_id",
            stable_id(
                "repeated_measures_receiver_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "contrast_name": self.contrast_name,
            "design_id": self.design_id,
            "effect_ids": [effect.effect_id for effect in self.feature_effects],
            "feature_ids": list(self.feature_ids),
            "receiver": self.receiver,
            "response_input_id": self.response_input_id,
            "sample_ids": list(self.sample_ids),
            "sample_values_digest": self.sample_values_digest,
        }

    @property
    def method(self) -> str:
        return _METHOD

    @property
    def uncertainty_semantics(self) -> str:
        return _UNCERTAINTY

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    def _require_intact(self) -> None:
        try:
            features = _names(self.feature_ids, field_name="feature_ids")
            samples = _names(
                self.sample_ids, field_name="sample_ids", allow_empty=True
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and len(self.feature_effects) == len(features)
                and tuple(effect.feature_id for effect in self.feature_effects)
                == features
                and all(
                    effect.effect_id
                    == stable_id(
                        "repeated_measures_receiver_effect",
                        effect._identity_payload(
                            response_input_id=self.response_input_id
                        ),
                        schema_version=_SCHEMA_VERSION,
                    )
                    for effect in self.feature_effects
                )
                and self.artifact_id
                == stable_id(
                    "repeated_measures_receiver_result",
                    self._identity_payload(),
                    schema_version=_SCHEMA_VERSION,
                )
                and len(samples) == len(self.sample_ids)
            )
        except Exception as error:
            raise ContractError(
                "Repeated-measures receiver result failed integrity validation",
                code="repeated_measures_receiver_result_integrity_violation",
                field="artifact_id",
                remediation="Refit the receiver effect from intact inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated-measures receiver result failed integrity validation",
                code="repeated_measures_receiver_result_integrity_violation",
                field="artifact_id",
                remediation="Refit the receiver effect from intact inputs",
            )

    def effect_for(self, feature_id: str) -> RepeatedMeasuresFeatureEffect:
        """Return one typed feature effect by its frozen feature identifier."""

        self._require_intact()
        try:
            index = self.feature_ids.index(feature_id)
        except ValueError as error:
            raise KeyError(feature_id) from error
        return self.feature_effects[index]

    def to_frame(self) -> pd.DataFrame:
        """Return all feature effects with result-level lineage columns."""

        self._require_intact()
        rows = []
        for effect in self.feature_effects:
            rows.append(
                {
                    "artifact_id": self.artifact_id,
                    "response_input_id": self.response_input_id,
                    "design_id": self.design_id,
                    "contrast": self.contrast_name,
                    "receiver": self.receiver,
                    **effect.as_record(),
                }
            )
        return pd.DataFrame.from_records(rows)


@dataclass(frozen=True, slots=True)
class _Support:
    n_subject_clusters: int
    n_contrast_subject_clusters: int
    n_context_subjects_min: int
    n_repeated_subject_clusters: int
    n_complete_contrast_subjects: int


def _support(
    design: FrozenRepeatedMeasuresDesign,
    usable_rows: np.ndarray,
) -> _Support:
    subjects = np.asarray(design.cell_subject_ids, dtype=object)[usable_rows]
    contexts = np.asarray(design.cell_context_ids, dtype=object)[usable_rows]
    subject_sets = {
        context_id: set(subjects[contexts == context_id].tolist())
        for context_id in design.contrast_context_ids
    }
    contrast_subjects = set().union(*subject_sets.values())
    complete = set.intersection(*subject_sets.values()) if subject_sets else set()
    repeated = sum(
        sum(subject in subject_sets[context_id] for context_id in subject_sets) > 1
        for subject in contrast_subjects
    )
    return _Support(
        n_subject_clusters=len(set(subjects.tolist())),
        n_contrast_subject_clusters=len(contrast_subjects),
        n_context_subjects_min=min(map(len, subject_sets.values()), default=0),
        n_repeated_subject_clusters=repeated,
        n_complete_contrast_subjects=len(complete),
    )


def _feature_effect(
    *,
    design: FrozenRepeatedMeasuresDesign,
    response_input_id: str,
    feature_id: str,
    cell_values: np.ndarray,
    n_input_samples: int,
) -> RepeatedMeasuresFeatureEffect:
    usable_rows = np.flatnonzero(np.isfinite(cell_values))
    support = _support(design, usable_rows)
    matrix = design.design_matrix[usable_rows]
    response = cell_values[usable_rows]
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    condition_number = float(np.linalg.cond(matrix)) if matrix.size else math.inf
    residual_df = len(response) - rank
    cluster_df = support.n_subject_clusters - 1
    reason = design.reason_code if not design.estimable else None
    if (
        reason is None
        and support.n_context_subjects_min < design.spec.min_subjects_per_context
    ):
        reason = "insufficient_subjects_per_context"
    if (
        reason is None
        and support.n_contrast_subject_clusters < design.spec.min_subject_clusters
    ):
        reason = "insufficient_subject_clusters"
    if reason is None and rank < matrix.shape[1]:
        reason = "rank_deficient_feature_design"
    if reason is None and not _vector_estimable(matrix, design.contrast_vector):
        reason = "feature_contrast_not_estimable"
    if reason is None and (
        not math.isfinite(condition_number)
        or condition_number > design.spec.max_condition_number
    ):
        reason = "ill_conditioned_feature_design"
    if reason is None and residual_df < 1:
        reason = "insufficient_residual_degrees_of_freedom"
    effect = math.nan
    standard_error = math.nan
    precision = math.nan
    diagnostics: tuple[str, ...] = ()
    if reason is None:
        gram_inverse = np.linalg.inv(matrix.T @ matrix)
        coefficients = gram_inverse @ matrix.T @ response
        residual = response - matrix @ coefficients
        subjects = np.asarray(design.cell_subject_ids, dtype=object)[usable_rows]
        meat = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
        for subject in sorted(set(subjects.tolist())):
            selected = subjects == subject
            score = matrix[selected].T @ residual[selected]
            meat += np.outer(score, score)
        correction = (
            support.n_subject_clusters / (support.n_subject_clusters - 1)
        ) * ((len(response) - 1) / residual_df)
        covariance = correction * gram_inverse @ meat @ gram_inverse
        variance = float(
            design.contrast_vector @ covariance @ design.contrast_vector
        )
        effect = float(design.contrast_vector @ coefficients)
        if (
            not math.isfinite(effect)
            or not math.isfinite(variance)
            or variance < -1.0e-12
        ):
            reason = "invalid_subject_cluster_estimate"
            effect = math.nan
        else:
            standard_error = math.sqrt(max(0.0, variance))
            if standard_error > 0:
                candidate_precision = 1.0 / standard_error**2
                if math.isfinite(candidate_precision):
                    precision = candidate_precision
                else:
                    diagnostics = ("non_finite_raw_precision",)
            else:
                diagnostics = ("zero_cluster_robust_standard_error",)

    status = "exploratory" if reason is None else "not_estimable"
    base = RepeatedMeasuresFeatureEffect(
        feature_id=feature_id,
        effect=effect,
        diagnostic_standard_error=standard_error,
        raw_precision=precision,
        n_input_samples=n_input_samples,
        n_model_cells=len(response),
        n_subject_clusters=support.n_subject_clusters,
        n_contrast_subject_clusters=support.n_contrast_subject_clusters,
        n_context_subjects_min=support.n_context_subjects_min,
        n_repeated_subject_clusters=support.n_repeated_subject_clusters,
        n_complete_contrast_subjects=support.n_complete_contrast_subjects,
        design_rank=rank,
        residual_df=residual_df if residual_df >= 0 else None,
        cluster_df=cluster_df if cluster_df >= 0 else None,
        condition_number=condition_number,
        status=status,
        reason_code=reason,
        diagnostic_codes=diagnostics,
        effect_id="pending",
    )
    effect_id = stable_id(
        "repeated_measures_receiver_effect",
        base._identity_payload(response_input_id=response_input_id),
        schema_version=_SCHEMA_VERSION,
    )
    return replace(base, effect_id=effect_id)


def fit_repeated_measures_receiver_effect(
    design: FrozenRepeatedMeasuresDesign,
    sample_values: np.ndarray,
    *,
    sample_ids: Sequence[str],
    feature_ids: Sequence[str],
    receiver: str,
) -> RepeatedMeasuresReceiverEffect:
    """Fit exploratory feature effects using subject as the sandwich cluster."""

    if not isinstance(design, FrozenRepeatedMeasuresDesign):
        raise TypeError("design must be a FrozenRepeatedMeasuresDesign")
    design._require_intact()
    samples = _names(sample_ids, field_name="sample_ids", allow_empty=True)
    features = _names(feature_ids, field_name="feature_ids")
    if not isinstance(receiver, str) or not receiver or receiver != receiver.strip():
        raise ValueError("receiver must be a canonical non-empty string")
    values = np.asarray(sample_values, dtype=np.float64)
    if values.shape != (len(samples), len(features)):
        raise ValueError("sample_values must align with sample_ids x feature_ids")
    if np.any(np.isinf(values)):
        raise ValueError("sample_values cannot contain infinity")
    unknown = set(samples).difference(design.sample_ids)
    if unknown:
        raise ValueError(
            "sample_values contain samples outside the frozen design: "
            + ",".join(sorted(unknown))
        )
    input_by_sample = {sample: index for index, sample in enumerate(samples)}
    aligned: np.ndarray = np.full(
        (len(design.sample_ids), len(features)), np.nan, dtype=np.float64
    )
    for output_index, sample_id in enumerate(design.sample_ids):
        input_index = input_by_sample.get(sample_id)
        if input_index is not None:
            aligned[output_index] = values[input_index]
    values_digest = _numeric_digest(aligned)
    response_input_id = stable_id(
        "repeated_measures_receiver_input",
        {
            "design_id": design.design_id,
            "feature_ids": list(features),
            "receiver": receiver,
            "sample_ids": list(design.sample_ids),
            "sample_values_digest": values_digest,
        },
        schema_version=_SCHEMA_VERSION,
        digest_length=64,
    )

    n_cells = len(design.cell_ids)
    cell_values: np.ndarray = np.full(
        (n_cells, len(features)), np.nan, dtype=np.float64
    )
    for cell_index in range(n_cells):
        selected = design.sample_cell_indices == cell_index
        cell_samples = aligned[selected]
        finite = np.isfinite(cell_samples)
        counts = finite.sum(axis=0)
        sums = np.where(finite, cell_samples, 0.0).sum(axis=0)
        observed = counts > 0
        cell_values[cell_index, observed] = sums[observed] / counts[observed]

    feature_effects = tuple(
        _feature_effect(
            design=design,
            response_input_id=response_input_id,
            feature_id=feature_id,
            cell_values=cell_values[:, feature_index],
            n_input_samples=int(np.isfinite(aligned[:, feature_index]).sum()),
        )
        for feature_index, feature_id in enumerate(features)
    )
    return RepeatedMeasuresReceiverEffect._from_fit(
        design=design,
        receiver=receiver,
        sample_ids=tuple(design.sample_ids),
        feature_ids=features,
        sample_values_digest=values_digest,
        response_input_id=response_input_id,
        feature_effects=feature_effects,
    )


__all__ = [
    "RepeatedMeasuresFeatureEffect",
    "RepeatedMeasuresReceiverEffect",
    "fit_repeated_measures_receiver_effect",
]
