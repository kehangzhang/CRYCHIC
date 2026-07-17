"""Fold lineage for repeated-measures receiver responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from crychic.core import ContractError, stable_id
from crychic.design import (
    FrozenDesignApplication,
    FrozenDesignEncoder,
    FrozenRepeatedMeasuresDesign,
    RepeatedMeasuresDesignSpec,
    freeze_repeated_measures_design,
)
from crychic.pseudobulk import PseudobulkDataset

from .fold import (
    _aligned_names,
    _canonical_name_key,
    _extract_receiver_response,
    _immutable_array,
    _names,
    _numeric_digest,
    _row_payload,
    _scope_name,
)
from .repeated_cr2 import (
    RepeatedMeasuresCR2ReceiverEffect,
    fit_repeated_measures_cr2_receiver_effect,
)
from .repeated_measures import (
    RepeatedMeasuresReceiverEffect,
    fit_repeated_measures_receiver_effect,
)
from .repeated_measures import (
    _numeric_digest as _repeated_numeric_digest,
)

_TRAINING_PRODUCER_MARKER = "crychic.response.repeated_fold.v1"
_APPLICATION_PRODUCER_MARKER = "crychic.response.repeated_fold_application.v1"
_METHOD = "formula_ols_subject_cluster_cr1_exploratory_v1"
_CR2_METHOD = "subject_equal_repeated_measures_cr2_outer_fold_v1"
_CR2_INPUT_BACKEND = "subject_equal_wls_cluster_cr2_v1"
_SCHEMA_VERSION = "1"

_RepeatedReceiverEffect: TypeAlias = (
    RepeatedMeasuresReceiverEffect | RepeatedMeasuresCR2ReceiverEffect
)


def _effect_vectors(
    effect: _RepeatedReceiverEffect,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimates = _immutable_array(
        np.asarray([item.effect for item in effect.feature_effects], dtype=np.float64)
    )
    if isinstance(effect, RepeatedMeasuresCR2ReceiverEffect):
        standard_error_values = [item.standard_error for item in effect.feature_effects]
    else:
        standard_error_values = [
            item.diagnostic_standard_error for item in effect.feature_effects
        ]
    standard_errors = _immutable_array(
        np.asarray(standard_error_values, dtype=np.float64)
    )
    raw_precision = _immutable_array(
        np.asarray(
            [item.raw_precision for item in effect.feature_effects], dtype=np.float64
        )
    )
    return estimates, standard_errors, raw_precision


def _overall_status(
    design: FrozenRepeatedMeasuresDesign,
    effect: _RepeatedReceiverEffect,
) -> tuple[str, str | None]:
    if not design.estimable:
        return "not_estimable", design.reason_code or "repeated_design_not_estimable"
    if isinstance(effect, RepeatedMeasuresCR2ReceiverEffect):
        if any(item.formal_backend_eligible for item in effect.feature_effects):
            return "ok", None
        reasons = {
            item.reason_code for item in effect.feature_effects if item.reason_code
        }
        reason = next(iter(reasons)) if len(reasons) == 1 else None
        return "not_estimable", reason or "repeated_cr2_receiver_effect_not_estimable"
    if any(item.status == "exploratory" for item in effect.feature_effects):
        return "exploratory", None
    reasons = {item.reason_code for item in effect.feature_effects if item.reason_code}
    reason = next(iter(reasons)) if len(reasons) == 1 else None
    return "not_estimable", reason or "repeated_receiver_effect_not_estimable"


def _effect_method(effect: _RepeatedReceiverEffect) -> str:
    return (
        _CR2_METHOD
        if isinstance(effect, RepeatedMeasuresCR2ReceiverEffect)
        else _METHOD
    )


def _expected_effect_input_id(
    effect: _RepeatedReceiverEffect,
    *,
    design: FrozenRepeatedMeasuresDesign,
    receiver: str,
    feature_ids: tuple[str, ...],
    sample_values_digest: str,
) -> str:
    if isinstance(effect, RepeatedMeasuresCR2ReceiverEffect):
        return str(
            stable_id(
                "repeated_measures_cr2_receiver_input",
                {
                    "backend": _CR2_INPUT_BACKEND,
                    "design_id": design.design_id,
                    "feature_ids": list(feature_ids),
                    "receiver": receiver,
                    "sample_ids": list(design.sample_ids),
                    "sample_values_digest": sample_values_digest,
                },
                schema_version="1.0.0",
                digest_length=64,
            )
        )
    return str(
        stable_id(
            "repeated_measures_receiver_input",
            {
                "design_id": design.design_id,
                "feature_ids": list(feature_ids),
                "receiver": receiver,
                "sample_ids": list(design.sample_ids),
                "sample_values_digest": sample_values_digest,
            },
            schema_version="1.0.0",
            digest_length=64,
        )
    )


def _aligned_effect_input(
    design: FrozenRepeatedMeasuresDesign,
    *,
    sample_ids: tuple[str, ...],
    sample_values: np.ndarray,
    n_features: int,
) -> np.ndarray:
    input_by_sample = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    aligned: np.ndarray = np.full(
        (len(design.sample_ids), n_features), np.nan, dtype=np.float64
    )
    for index, sample_id in enumerate(design.sample_ids):
        input_index = input_by_sample.get(sample_id)
        if input_index is not None:
            aligned[index] = sample_values[input_index]
    return aligned


@dataclass(frozen=True, slots=True, init=False)
class RepeatedMeasuresFoldResponseArtifact:
    """Producer-owned repeated response with exact fold and sample lineage."""

    receiver: str
    contrast_name: str
    fold_id: str
    training_input_digest: str
    encoder_id: str
    context_regressor_id: str
    nuisance_design_id: str
    training_sample_manifest_digest: str
    training_design_digest: str
    coefficient_contrast_digest: str
    reparameterization_digest: str
    context_keys: tuple[str, ...]
    feature_ids: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    training_sample_subject_ids: tuple[str, ...]
    training_sample_context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    missing_sample_ids: tuple[str, ...]
    sample_values: np.ndarray
    effect: np.ndarray
    standard_error: np.ndarray
    raw_precision: np.ndarray
    response_input_digest: str
    sample_values_digest: str
    effect_digest: str
    standard_error_digest: str
    raw_precision_digest: str
    repeated_design: FrozenRepeatedMeasuresDesign
    repeated_effect: _RepeatedReceiverEffect
    repeated_design_id: str
    repeated_effect_artifact_id: str
    method: str
    value_scale: str
    min_subjects_per_context: int
    min_subject_clusters: int
    n_model_samples: int
    n_model_subjects: int
    residual_df: None
    status: str
    reason_code: str | None
    artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "RepeatedMeasuresFoldResponseArtifact is producer-owned; "
            "use fit_repeated_measures_fold_response()"
        )

    @classmethod
    def _from_fit(
        cls,
        *,
        encoder: FrozenDesignEncoder,
        design: FrozenRepeatedMeasuresDesign,
        repeated_effect: _RepeatedReceiverEffect,
        receiver: str,
        fold_id: str,
        training_input_digest: str,
        sample_ids: tuple[str, ...],
        sample_subject_ids: tuple[str, ...],
        sample_context_ids: tuple[str, ...],
        missing_sample_ids: tuple[str, ...],
        sample_values: np.ndarray,
        feature_ids: tuple[str, ...],
    ) -> RepeatedMeasuresFoldResponseArtifact:
        values = _immutable_array(sample_values)
        estimates, standard_errors, raw_precision = _effect_vectors(repeated_effect)
        status, reason = _overall_status(design, repeated_effect)
        sample_values_digest = _numeric_digest(values)
        effect_digest = _numeric_digest(estimates)
        standard_error_digest = _numeric_digest(standard_errors)
        raw_precision_digest = _numeric_digest(raw_precision)
        response_input_digest = stable_id(
            "repeated_measures_fold_response_input",
            {
                "encoder_id": encoder.encoder_id,
                "feature_ids": list(feature_ids),
                "missing_sample_ids": list(missing_sample_ids),
                "receiver": receiver,
                "repeated_design_id": design.design_id,
                "repeated_effect_artifact_id": repeated_effect.artifact_id,
                "rows": _row_payload(
                    sample_ids, sample_subject_ids, sample_context_ids
                ),
                "sample_values_digest": sample_values_digest,
                "training_input_digest": training_input_digest,
            },
            schema_version=_SCHEMA_VERSION,
            digest_length=64,
        )
        self = object.__new__(cls)
        values_by_name: dict[str, Any] = {
            "receiver": receiver,
            "contrast_name": encoder.contrast.name,
            "fold_id": fold_id,
            "training_input_digest": training_input_digest,
            "encoder_id": encoder.encoder_id,
            "context_regressor_id": encoder.context_regressor_id,
            "nuisance_design_id": encoder.nuisance_design_id,
            "training_sample_manifest_digest": (
                encoder.training_sample_manifest_digest
            ),
            "training_design_digest": encoder.training_design_digest,
            "coefficient_contrast_digest": encoder.coefficient_contrast_digest,
            "reparameterization_digest": encoder.reparameterization_digest,
            "context_keys": encoder.context_keys,
            "feature_ids": feature_ids,
            "training_sample_ids": encoder.training_sample_ids,
            "training_sample_subject_ids": encoder.training_sample_subject_ids,
            "training_sample_context_ids": encoder.training_sample_context_ids,
            "training_subject_ids": encoder.training_subject_ids,
            "sample_ids": sample_ids,
            "sample_subject_ids": sample_subject_ids,
            "sample_context_ids": sample_context_ids,
            "subject_ids": tuple(sorted(set(sample_subject_ids))),
            "missing_sample_ids": tuple(sorted(missing_sample_ids)),
            "sample_values": values,
            "effect": estimates,
            "standard_error": standard_errors,
            "raw_precision": raw_precision,
            "response_input_digest": response_input_digest,
            "sample_values_digest": sample_values_digest,
            "effect_digest": effect_digest,
            "standard_error_digest": standard_error_digest,
            "raw_precision_digest": raw_precision_digest,
            "repeated_design": design,
            "repeated_effect": repeated_effect,
            "repeated_design_id": design.design_id,
            "repeated_effect_artifact_id": repeated_effect.artifact_id,
            "method": _effect_method(repeated_effect),
            "value_scale": "log1p_cpm",
            "min_subjects_per_context": design.spec.min_subjects_per_context,
            "min_subject_clusters": design.spec.min_subject_clusters,
            "n_model_samples": len(design.cell_ids),
            "n_model_subjects": design.n_subject_clusters,
            "residual_df": None,
            "status": status,
            "reason_code": reason,
            "_producer_marker": _TRAINING_PRODUCER_MARKER,
        }
        for name, value in values_by_name.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "artifact_id",
            stable_id(
                "repeated_measures_fold_response",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    @property
    def formal_inference_allowed(self) -> bool:
        return False

    @property
    def is_cr1_exploratory(self) -> bool:
        """Whether this is the preserved legacy CR1 diagnostic producer."""

        self._require_intact()
        return self.method == _METHOD

    @property
    def cr2_backend_eligible(self) -> bool:
        """Whether the response has feature support from the strict CR2 backend."""

        self._require_intact()
        return self.method == _CR2_METHOD and self.status == "ok"

    @property
    def observed_status(self) -> str:
        """Return the backend-specific usable training status."""

        return "exploratory" if self.method == _METHOD else "ok"

    def _identity_payload(self) -> dict[str, object]:
        return {
            "coefficient_contrast_digest": self.coefficient_contrast_digest,
            "context_regressor_id": self.context_regressor_id,
            "context_keys": list(self.context_keys),
            "contrast_name": self.contrast_name,
            "effect_digest": self.effect_digest,
            "encoder_id": self.encoder_id,
            "fold_id": self.fold_id,
            "method": self.method,
            "missing_sample_ids": list(self.missing_sample_ids),
            "n_model_samples": self.n_model_samples,
            "n_model_subjects": self.n_model_subjects,
            "nuisance_design_id": self.nuisance_design_id,
            "raw_precision_digest": self.raw_precision_digest,
            "reason_code": self.reason_code,
            "reparameterization_digest": self.reparameterization_digest,
            "repeated_design_id": self.repeated_design_id,
            "repeated_effect_artifact_id": self.repeated_effect_artifact_id,
            "response_input_digest": self.response_input_digest,
            "standard_error_digest": self.standard_error_digest,
            "status": self.status,
            "training_design_digest": self.training_design_digest,
            "training_input_digest": self.training_input_digest,
            "training_sample_manifest_digest": self.training_sample_manifest_digest,
            "training_rows": _row_payload(
                self.training_sample_ids,
                self.training_sample_subject_ids,
                self.training_sample_context_ids,
            ),
            "training_subject_ids": list(self.training_subject_ids),
            "value_scale": self.value_scale,
        }

    def _require_intact(self) -> None:
        try:
            self.repeated_design._require_intact()
            self.repeated_effect._require_intact()
            samples = _names(self.sample_ids, field_name="sample_ids", allow_empty=True)
            sample_subjects = _aligned_names(
                self.sample_subject_ids,
                length=len(samples),
                field_name="sample_subject_ids",
            )
            sample_contexts = _aligned_names(
                self.sample_context_ids,
                length=len(samples),
                field_name="sample_context_ids",
            )
            features = _names(self.feature_ids, field_name="feature_ids")
            missing = tuple(sorted(self.missing_sample_ids))
            arrays = (
                self.sample_values,
                self.effect,
                self.standard_error,
                self.raw_precision,
            )
            if self.sample_values.shape != (len(samples), len(features)) or any(
                array.shape != (len(features),) for array in arrays[1:]
            ):
                raise ValueError("repeated response arrays are misaligned")
            if any(array.flags.writeable for array in arrays):
                raise ValueError("repeated response arrays must be immutable")
            if np.any(~np.isfinite(self.sample_values)):
                raise ValueError("observed repeated response rows must be finite")
            if samples != tuple(sorted(samples, key=_canonical_name_key)):
                raise ValueError("repeated response sample rows are not canonical")
            if self.subject_ids != tuple(sorted(set(sample_subjects))):
                raise ValueError("repeated response subjects are inconsistent")
            training_mapping = dict(
                zip(
                    self.training_sample_ids,
                    zip(
                        self.training_sample_subject_ids,
                        self.training_sample_context_ids,
                        strict=True,
                    ),
                    strict=True,
                )
            )
            design_mapping = {
                sample_id: (
                    subject_id,
                    self.repeated_design.cell_context_ids[int(cell_index)],
                )
                for sample_id, subject_id, cell_index in zip(
                    self.repeated_design.sample_ids,
                    self.repeated_design.sample_subject_ids,
                    self.repeated_design.sample_cell_indices,
                    strict=True,
                )
            }
            if design_mapping != training_mapping:
                raise ValueError(
                    "repeated design rows differ from encoder training rows"
                )
            if any(
                training_mapping[sample] != (subject, context)
                for sample, subject, context in zip(
                    samples, sample_subjects, sample_contexts, strict=True
                )
            ):
                raise ValueError("repeated response rows differ from training lineage")
            if set(samples).intersection(missing) or set(samples).union(missing) != set(
                self.training_sample_ids
            ):
                raise ValueError("repeated response coverage is inconsistent")
            expected_status, expected_reason = _overall_status(
                self.repeated_design, self.repeated_effect
            )
            effect, standard_error, raw_precision = _effect_vectors(
                self.repeated_effect
            )
            aligned = _aligned_effect_input(
                self.repeated_design,
                sample_ids=samples,
                sample_values=self.sample_values,
                n_features=len(features),
            )
            effect_values_digest = _repeated_numeric_digest(aligned)
            expected_effect_input_id = _expected_effect_input_id(
                self.repeated_effect,
                design=self.repeated_design,
                receiver=self.receiver,
                feature_ids=features,
                sample_values_digest=effect_values_digest,
            )
            sample_values_digest = _numeric_digest(self.sample_values)
            effect_digest = _numeric_digest(self.effect)
            standard_error_digest = _numeric_digest(self.standard_error)
            raw_precision_digest = _numeric_digest(self.raw_precision)
            response_input_digest = stable_id(
                "repeated_measures_fold_response_input",
                {
                    "encoder_id": self.encoder_id,
                    "feature_ids": list(features),
                    "missing_sample_ids": list(missing),
                    "receiver": self.receiver,
                    "repeated_design_id": self.repeated_design_id,
                    "repeated_effect_artifact_id": (self.repeated_effect_artifact_id),
                    "rows": _row_payload(samples, sample_subjects, sample_contexts),
                    "sample_values_digest": sample_values_digest,
                    "training_input_digest": self.training_input_digest,
                },
                schema_version=_SCHEMA_VERSION,
                digest_length=64,
            )
            expected_id = stable_id(
                "repeated_measures_fold_response",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _TRAINING_PRODUCER_MARKER
                and self.method == _effect_method(self.repeated_effect)
                and self.value_scale == "log1p_cpm"
                and self.residual_df is None
                and self.status == expected_status
                and self.reason_code == expected_reason
                and self.repeated_design_id == self.repeated_design.design_id
                and self.repeated_effect_artifact_id == self.repeated_effect.artifact_id
                and self.repeated_effect.design_id == self.repeated_design.design_id
                and self.repeated_effect.response_input_id == expected_effect_input_id
                and self.repeated_effect.feature_ids == features
                and self.repeated_effect.receiver == self.receiver
                and self.repeated_effect.sample_ids == self.repeated_design.sample_ids
                and np.array_equal(self.effect, effect, equal_nan=True)
                and np.array_equal(self.standard_error, standard_error, equal_nan=True)
                and np.array_equal(self.raw_precision, raw_precision, equal_nan=True)
                and sample_values_digest == self.sample_values_digest
                and effect_digest == self.effect_digest
                and standard_error_digest == self.standard_error_digest
                and raw_precision_digest == self.raw_precision_digest
                and response_input_digest == self.response_input_digest
                and expected_id == self.artifact_id
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Repeated-measures fold response failed integrity validation",
                code="repeated_fold_response_integrity_violation",
                field="artifact_id",
                remediation="Refit the repeated response from intact fold parents",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated-measures fold response failed integrity validation",
                code="repeated_fold_response_integrity_violation",
                field="artifact_id",
                remediation="Refit the repeated response from intact fold parents",
            )

    def require_compatible(self, encoder: FrozenDesignEncoder) -> None:
        """Validate exact frozen encoder parentage."""

        if not isinstance(encoder, FrozenDesignEncoder):
            raise TypeError("encoder must be a FrozenDesignEncoder")
        encoder.to_dict()
        self._require_intact()
        expected = (
            encoder.encoder_id,
            encoder.context_regressor_id,
            encoder.nuisance_design_id,
            encoder.training_sample_manifest_digest,
            encoder.training_design_digest,
            encoder.coefficient_contrast_digest,
            encoder.reparameterization_digest,
            encoder.contrast.name,
            encoder.context_keys,
            encoder.training_sample_ids,
            encoder.training_sample_subject_ids,
            encoder.training_sample_context_ids,
            encoder.training_subject_ids,
        )
        observed = (
            self.encoder_id,
            self.context_regressor_id,
            self.nuisance_design_id,
            self.training_sample_manifest_digest,
            self.training_design_digest,
            self.coefficient_contrast_digest,
            self.reparameterization_digest,
            self.contrast_name,
            self.context_keys,
            self.training_sample_ids,
            self.training_sample_subject_ids,
            self.training_sample_context_ids,
            self.training_subject_ids,
        )
        if observed != expected:
            raise ContractError(
                "Repeated fold response does not match its frozen encoder",
                code="repeated_fold_response_scope_mismatch",
                field="artifact_id",
                remediation="Use the response fitted from this exact encoder",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"artifact_id": self.artifact_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, init=False)
class RepeatedMeasuresFoldResponseApplication:
    """Held-out repeated-path receiver rows extracted without refitting."""

    training_response_id: str
    design_application_id: str
    encoder_id: str
    context_regressor_id: str
    nuisance_design_id: str
    receiver: str
    contrast_name: str
    fold_id: str
    feature_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    missing_sample_ids: tuple[str, ...]
    sample_values: np.ndarray
    sample_values_digest: str
    response_input_digest: str
    status: str
    reason_code: str | None
    value_scale: str
    application_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "RepeatedMeasuresFoldResponseApplication is producer-owned; "
            "use apply_repeated_measures_fold_response()"
        )

    @classmethod
    def _from_application(
        cls,
        *,
        training_response: RepeatedMeasuresFoldResponseArtifact,
        design_application: FrozenDesignApplication,
        sample_values: np.ndarray,
        missing_sample_ids: tuple[str, ...],
        status: str,
        reason_code: str | None,
    ) -> RepeatedMeasuresFoldResponseApplication:
        values = _immutable_array(sample_values)
        samples = tuple(design_application.sample_ids)
        sample_subjects = tuple(design_application.sample_subject_ids)
        sample_contexts = tuple(design_application.sample_context_ids)
        missing = tuple(sorted(missing_sample_ids))
        values_digest = _numeric_digest(values)
        response_input_digest = stable_id(
            "repeated_measures_fold_response_application_input",
            {
                "design_application_id": design_application.application_id,
                "feature_ids": list(training_response.feature_ids),
                "missing_sample_ids": list(missing),
                "rows": _row_payload(samples, sample_subjects, sample_contexts),
                "sample_values_digest": values_digest,
                "training_response_id": training_response.artifact_id,
            },
            schema_version=_SCHEMA_VERSION,
            digest_length=64,
        )
        self = object.__new__(cls)
        values_by_name: dict[str, Any] = {
            "training_response_id": training_response.artifact_id,
            "design_application_id": design_application.application_id,
            "encoder_id": training_response.encoder_id,
            "context_regressor_id": training_response.context_regressor_id,
            "nuisance_design_id": training_response.nuisance_design_id,
            "receiver": training_response.receiver,
            "contrast_name": training_response.contrast_name,
            "fold_id": training_response.fold_id,
            "feature_ids": training_response.feature_ids,
            "sample_ids": samples,
            "sample_subject_ids": sample_subjects,
            "sample_context_ids": sample_contexts,
            "subject_ids": tuple(sorted(set(sample_subjects))),
            "missing_sample_ids": missing,
            "sample_values": values,
            "sample_values_digest": values_digest,
            "response_input_digest": response_input_digest,
            "status": status,
            "reason_code": reason_code,
            "value_scale": training_response.value_scale,
            "_producer_marker": _APPLICATION_PRODUCER_MARKER,
        }
        for name, value in values_by_name.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "application_id",
            stable_id(
                "repeated_measures_fold_response_application",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "context_regressor_id": self.context_regressor_id,
            "contrast_name": self.contrast_name,
            "design_application_id": self.design_application_id,
            "encoder_id": self.encoder_id,
            "fold_id": self.fold_id,
            "nuisance_design_id": self.nuisance_design_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "response_input_digest": self.response_input_digest,
            "status": self.status,
            "training_response_id": self.training_response_id,
            "value_scale": self.value_scale,
        }

    def _require_intact(self) -> None:
        try:
            samples = _names(self.sample_ids, field_name="sample_ids")
            sample_subjects = _aligned_names(
                self.sample_subject_ids,
                length=len(samples),
                field_name="sample_subject_ids",
            )
            sample_contexts = _aligned_names(
                self.sample_context_ids,
                length=len(samples),
                field_name="sample_context_ids",
            )
            features = _names(self.feature_ids, field_name="feature_ids")
            missing = tuple(sorted(self.missing_sample_ids))
            missing_positions = [samples.index(sample_id) for sample_id in missing]
            observed_positions = [
                index for index in range(len(samples)) if index not in missing_positions
            ]
            if self.sample_values.shape != (len(samples), len(features)):
                raise ValueError("held-out repeated response matrix is misaligned")
            if self.sample_values.flags.writeable:
                raise ValueError("held-out repeated response matrix is mutable")
            if (
                missing_positions
                and not np.isnan(self.sample_values[missing_positions]).all()
            ):
                raise ValueError("missing held-out repeated rows must remain NaN")
            if observed_positions and np.any(
                ~np.isfinite(self.sample_values[observed_positions])
            ):
                raise ValueError("observed held-out repeated rows must be finite")
            if samples != tuple(sorted(samples, key=_canonical_name_key)):
                raise ValueError("held-out repeated response rows are not canonical")
            if self.subject_ids != tuple(sorted(set(sample_subjects))):
                raise ValueError("held-out repeated subjects are inconsistent")
            if not set(missing).issubset(samples):
                raise ValueError("held-out repeated missing rows are invalid")
            if self.status not in {"ok", "not_estimable"} or (
                (self.status == "ok") == (self.reason_code is not None)
            ):
                raise ValueError("held-out repeated response status is inconsistent")
            if self.status == "ok" and missing:
                raise ValueError("observed held-out repeated response cannot omit rows")
            values_digest = _numeric_digest(self.sample_values)
            response_input_digest = stable_id(
                "repeated_measures_fold_response_application_input",
                {
                    "design_application_id": self.design_application_id,
                    "feature_ids": list(features),
                    "missing_sample_ids": list(missing),
                    "rows": _row_payload(samples, sample_subjects, sample_contexts),
                    "sample_values_digest": values_digest,
                    "training_response_id": self.training_response_id,
                },
                schema_version=_SCHEMA_VERSION,
                digest_length=64,
            )
            expected_id = stable_id(
                "repeated_measures_fold_response_application",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _APPLICATION_PRODUCER_MARKER
                and values_digest == self.sample_values_digest
                and response_input_digest == self.response_input_digest
                and expected_id == self.application_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Repeated fold response application failed integrity validation",
                code="repeated_fold_response_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact repeated response parent",
            ) from error
        if not valid:
            raise ContractError(
                "Repeated fold response application failed integrity validation",
                code="repeated_fold_response_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact repeated response parent",
            )

    def require_compatible(
        self,
        training_response: RepeatedMeasuresFoldResponseArtifact,
        design_application: FrozenDesignApplication,
    ) -> None:
        if not isinstance(training_response, RepeatedMeasuresFoldResponseArtifact):
            raise TypeError(
                "training_response must be a RepeatedMeasuresFoldResponseArtifact"
            )
        if not isinstance(design_application, FrozenDesignApplication):
            raise TypeError("design_application must be a FrozenDesignApplication")
        training_response._require_intact()
        design_application.to_dict()
        self._require_intact()
        expected = (
            training_response.artifact_id,
            design_application.application_id,
            training_response.encoder_id,
            training_response.context_regressor_id,
            training_response.nuisance_design_id,
            training_response.feature_ids,
        )
        observed = (
            self.training_response_id,
            self.design_application_id,
            self.encoder_id,
            self.context_regressor_id,
            self.nuisance_design_id,
            self.feature_ids,
        )
        if observed != expected:
            raise ContractError(
                "Repeated fold response application does not match its parents",
                code="repeated_fold_response_application_scope_mismatch",
                field="application_id",
                remediation="Use the application produced from these exact parents",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"application_id": self.application_id, **self._identity_payload()}


def _validate_training_metadata(
    sample_metadata: pd.DataFrame,
    encoder: FrozenDesignEncoder,
) -> pd.DataFrame:
    required = {
        encoder.sample_key,
        encoder.subject_key,
        *encoder.context_keys,
        *encoder.covariates,
    }
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"sample_metadata is missing columns: {sorted(missing)}")
    table = sample_metadata.loc[:, sorted(required)].copy(deep=True)
    table[encoder.sample_key] = table[encoder.sample_key].astype(str)
    table[encoder.subject_key] = table[encoder.subject_key].astype(str)
    if table[encoder.sample_key].duplicated().any():
        raise ValueError("sample_metadata must contain one row per sample")
    if set(table[encoder.sample_key]) != set(encoder.training_sample_ids):
        raise ContractError(
            "Repeated design metadata does not exactly match encoder training rows",
            code="repeated_fold_response_scope_mismatch",
            field=encoder.sample_key,
            remediation="Pass only the physical outer-training sample metadata",
        )
    subject_by_sample = dict(
        zip(table[encoder.sample_key], table[encoder.subject_key], strict=True)
    )
    expected_subject_by_sample = dict(
        zip(
            encoder.training_sample_ids,
            encoder.training_sample_subject_ids,
            strict=True,
        )
    )
    if subject_by_sample != expected_subject_by_sample:
        raise ContractError(
            "Repeated design subjects differ from encoder training lineage",
            code="repeated_fold_response_scope_mismatch",
            field=encoder.subject_key,
            remediation="Use the sample metadata that fitted the frozen encoder",
        )
    return table


def fit_repeated_measures_fold_response(
    aggregate: PseudobulkDataset,
    encoder: FrozenDesignEncoder,
    sample_metadata: pd.DataFrame,
    *,
    receiver: str,
    fold_id: str,
    training_input_digest: str,
    min_subjects_per_context: int = 3,
    min_subject_clusters: int = 6,
) -> RepeatedMeasuresFoldResponseArtifact:
    """Fit a CR1-diagnostic repeated response under exact outer-fold lineage."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("aggregate must be a PseudobulkDataset")
    if not isinstance(encoder, FrozenDesignEncoder):
        raise TypeError("encoder must be a FrozenDesignEncoder")
    encoder.to_dict()
    receiver_name = _scope_name(receiver, field_name="receiver")
    fold = _scope_name(fold_id, field_name="fold_id")
    input_digest = _scope_name(
        training_input_digest, field_name="training_input_digest"
    )
    for field_name, value in (
        ("min_subjects_per_context", min_subjects_per_context),
        ("min_subject_clusters", min_subject_clusters),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{field_name} must be an integer >= 2")
    metadata = _validate_training_metadata(sample_metadata, encoder)
    design = freeze_repeated_measures_design(
        metadata,
        contrast=encoder.contrast,
        spec=RepeatedMeasuresDesignSpec(
            context_keys=encoder.context_keys,
            covariates=encoder.covariates,
            categorical_covariates=encoder.categorical_covariates,
            formula=encoder.formula,
            sample_key=encoder.sample_key,
            subject_key=encoder.subject_key,
            min_subjects_per_context=min_subjects_per_context,
            min_subject_clusters=min_subject_clusters,
        ),
    )
    extracted = _extract_receiver_response(
        aggregate,
        receiver=receiver_name,
        sample_ids=tuple(encoder.training_sample_ids),
        sample_subject_ids=tuple(encoder.training_sample_subject_ids),
        sample_context_ids=tuple(encoder.training_sample_context_ids),
        context_keys=encoder.context_keys,
    )
    feature_ids = _names(aggregate.feature_ids, field_name="feature_ids")
    repeated_effect = fit_repeated_measures_receiver_effect(
        design,
        extracted.values,
        sample_ids=extracted.sample_ids,
        feature_ids=feature_ids,
        receiver=receiver_name,
    )
    return RepeatedMeasuresFoldResponseArtifact._from_fit(
        encoder=encoder,
        design=design,
        repeated_effect=repeated_effect,
        receiver=receiver_name,
        fold_id=fold,
        training_input_digest=input_digest,
        sample_ids=extracted.sample_ids,
        sample_subject_ids=extracted.sample_subject_ids,
        sample_context_ids=extracted.sample_context_ids,
        missing_sample_ids=extracted.missing_sample_ids,
        sample_values=extracted.values,
        feature_ids=feature_ids,
    )


def fit_repeated_measures_cr2_fold_response(
    aggregate: PseudobulkDataset,
    encoder: FrozenDesignEncoder,
    sample_metadata: pd.DataFrame,
    *,
    receiver: str,
    fold_id: str,
    training_input_digest: str,
    min_subjects_per_context: int = 3,
    min_subject_clusters: int = 6,
) -> RepeatedMeasuresFoldResponseArtifact:
    """Fit a strict CR2 receiver response under exact outer-fold lineage."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("aggregate must be a PseudobulkDataset")
    if not isinstance(encoder, FrozenDesignEncoder):
        raise TypeError("encoder must be a FrozenDesignEncoder")
    encoder.to_dict()
    receiver_name = _scope_name(receiver, field_name="receiver")
    fold = _scope_name(fold_id, field_name="fold_id")
    input_digest = _scope_name(
        training_input_digest, field_name="training_input_digest"
    )
    for field_name, value in (
        ("min_subjects_per_context", min_subjects_per_context),
        ("min_subject_clusters", min_subject_clusters),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{field_name} must be an integer >= 2")
    metadata = _validate_training_metadata(sample_metadata, encoder)
    design = freeze_repeated_measures_design(
        metadata,
        contrast=encoder.contrast,
        spec=RepeatedMeasuresDesignSpec(
            context_keys=encoder.context_keys,
            covariates=encoder.covariates,
            categorical_covariates=encoder.categorical_covariates,
            formula=encoder.formula,
            sample_key=encoder.sample_key,
            subject_key=encoder.subject_key,
            min_subjects_per_context=min_subjects_per_context,
            min_subject_clusters=min_subject_clusters,
        ),
    )
    extracted = _extract_receiver_response(
        aggregate,
        receiver=receiver_name,
        sample_ids=tuple(encoder.training_sample_ids),
        sample_subject_ids=tuple(encoder.training_sample_subject_ids),
        sample_context_ids=tuple(encoder.training_sample_context_ids),
        context_keys=encoder.context_keys,
    )
    feature_ids = _names(aggregate.feature_ids, field_name="feature_ids")
    repeated_effect = fit_repeated_measures_cr2_receiver_effect(
        design,
        extracted.values,
        sample_ids=extracted.sample_ids,
        feature_ids=feature_ids,
        receiver=receiver_name,
    )
    return RepeatedMeasuresFoldResponseArtifact._from_fit(
        encoder=encoder,
        design=design,
        repeated_effect=repeated_effect,
        receiver=receiver_name,
        fold_id=fold,
        training_input_digest=input_digest,
        sample_ids=extracted.sample_ids,
        sample_subject_ids=extracted.sample_subject_ids,
        sample_context_ids=extracted.sample_context_ids,
        missing_sample_ids=extracted.missing_sample_ids,
        sample_values=extracted.values,
        feature_ids=feature_ids,
    )


def apply_repeated_measures_fold_response(
    aggregate: PseudobulkDataset,
    training_response: RepeatedMeasuresFoldResponseArtifact,
    design_application: FrozenDesignApplication,
) -> RepeatedMeasuresFoldResponseApplication:
    """Extract held-out receiver rows for an exploratory repeated model."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("aggregate must be a PseudobulkDataset")
    if not isinstance(training_response, RepeatedMeasuresFoldResponseArtifact):
        raise TypeError(
            "training_response must be a RepeatedMeasuresFoldResponseArtifact"
        )
    if not isinstance(design_application, FrozenDesignApplication):
        raise TypeError("design_application must be a FrozenDesignApplication")
    training_response._require_intact()
    design_application.to_dict()
    if (
        design_application.encoder_id,
        design_application.context_regressor_id,
        design_application.nuisance_design_id,
    ) != (
        training_response.encoder_id,
        training_response.context_regressor_id,
        training_response.nuisance_design_id,
    ):
        raise ContractError(
            "Held-out design does not match the repeated response encoder",
            code="repeated_fold_response_application_scope_mismatch",
            field="design_application_id",
            remediation="Use the held-out application from the response encoder",
        )
    if tuple(aggregate.feature_ids) != training_response.feature_ids:
        raise ContractError(
            "Held-out repeated response feature order differs from training",
            code="repeated_fold_response_application_feature_mismatch",
            field="feature_ids",
            remediation="Use the exact ordered training feature universe",
        )
    overlap = set(design_application.sample_subject_ids).intersection(
        training_response.training_subject_ids
    )
    if overlap:
        raise ValueError(
            "held-out repeated response overlaps training subjects: "
            + ", ".join(sorted(overlap))
        )
    extracted = _extract_receiver_response(
        aggregate,
        receiver=training_response.receiver,
        sample_ids=tuple(design_application.sample_ids),
        sample_subject_ids=tuple(design_application.sample_subject_ids),
        sample_context_ids=tuple(design_application.sample_context_ids),
        context_keys=training_response.context_keys,
    )
    observed_by_sample = {
        sample_id: extracted.values[index]
        for index, sample_id in enumerate(extracted.sample_ids)
    }
    aligned: np.ndarray = np.full(
        (len(design_application.sample_ids), len(training_response.feature_ids)),
        np.nan,
        dtype=np.float64,
    )
    for index, sample_id in enumerate(design_application.sample_ids):
        if sample_id in observed_by_sample:
            aligned[index] = observed_by_sample[sample_id]
    if design_application.status != "observed":
        status = "not_estimable"
        reason = "design_application_not_estimable:" + str(
            design_application.reason_code
        )
    elif training_response.status != training_response.observed_status:
        status = "not_estimable"
        reason = "training_response_not_estimable:" + str(training_response.reason_code)
    elif extracted.missing_sample_ids:
        status = "not_estimable"
        reason = "incomplete_heldout_receiver_response"
    else:
        status = "ok"
        reason = None
    return RepeatedMeasuresFoldResponseApplication._from_application(
        training_response=training_response,
        design_application=design_application,
        sample_values=aligned,
        missing_sample_ids=extracted.missing_sample_ids,
        status=status,
        reason_code=reason,
    )


__all__ = [
    "RepeatedMeasuresFoldResponseApplication",
    "RepeatedMeasuresFoldResponseArtifact",
    "apply_repeated_measures_fold_response",
    "fit_repeated_measures_cr2_fold_response",
    "fit_repeated_measures_fold_response",
]
