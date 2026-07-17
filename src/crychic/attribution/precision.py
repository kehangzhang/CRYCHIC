"""Fold-frozen precision transforms for receiver attribution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from crychic.core import ContractError, stable_id
from crychic.response import FoldGeneResponseArtifact
from crychic.response.repeated_fold import RepeatedMeasuresFoldResponseArtifact

_EXPLORATORY_METHOD = "winsorized_median_normalized_v2"
_STANDARDIZED_GUARDRAIL_METHOD = (
    "standardized_inverse_variance_with_low_df_equal_support_guardrail_v1"
)
_PRODUCER_MARKER = "crychic.attribution.precision.v3"
_TRANSFORM_SCHEMA_VERSION = "3"
_EXPLORATORY_LINEAGE_MODE = "exploratory_unparented_v2"
_RESPONSE_LINEAGE_MODE = "fold_gene_response_parented_v1"
_REPEATED_RESPONSE_LINEAGE_MODE = "repeated_measures_fold_response_parented_v1"
_REPEATED_CR2_RESPONSE_LINEAGE_MODE = "repeated_measures_cr2_fold_response_parented_v1"
_REPEATED_STANDARDIZED_METHOD = (
    "repeated_cr1_diagnostic_standardized_inverse_variance_v1"
)
_REPEATED_CR2_STANDARDIZED_METHOD = "repeated_cr2_standardized_inverse_variance_v1"
_LOW_DF_THRESHOLD = 4
_RAW_WEIGHT_MODE = "winsorized_raw_inverse_variance_v2"
_STANDARDIZED_WEIGHT_MODE = "winsorized_standardized_inverse_variance_v1"
_LOW_DF_WEIGHT_MODE = "equal_supported_low_residual_df_v1"
_UNKNOWN_DF_WEIGHT_MODE = "equal_supported_unknown_residual_df_v1"
_MISSING_SCALE_WEIGHT_MODE = "equal_supported_feature_scale_unavailable_v1"
_CALLER_SCALE_SOURCE = "caller_supplied_downstream_feature_scale_v1"
_MISSING_SCALE_SOURCE = "not_provided_v1"
_ARTIFACT_DF_SOURCE = "fold_gene_response_residual_df_v1"
_MISSING_DF_SOURCE = "not_estimable_fold_gene_response_v1"
_REPEATED_DF_SOURCE = "feature_specific_subject_cluster_cr1_v1"
_REPEATED_CR2_DF_SOURCE = "feature_specific_subject_cluster_cr2_v1"
_REPEATED_STANDARDIZED_WEIGHT_MODE = (
    "winsorized_repeated_cr1_standardized_inverse_variance_v1"
)
_REPEATED_CR2_STANDARDIZED_WEIGHT_MODE = (
    "winsorized_repeated_cr2_standardized_inverse_variance_v1"
)
_REPEATED_MISSING_SCALE_WEIGHT_MODE = (
    "equal_supported_repeated_cr1_feature_scale_unavailable_v1"
)
_REPEATED_CR2_MISSING_SCALE_WEIGHT_MODE = (
    "equal_supported_repeated_cr2_feature_scale_unavailable_v1"
)


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _scope_name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _immutable_vector(values: np.ndarray) -> np.ndarray:
    """Return a float64 vector backed by immutable bytes."""

    owned = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    result = cast(
        np.ndarray,
        np.frombuffer(owned.tobytes(order="C"), dtype="<f8").reshape(owned.shape),
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


def _validated_lineage(
    *,
    lineage_mode: str,
    response_artifact_id: str | None,
    training_row_manifest_id: str | None,
    training_subject_ids: Sequence[str],
    encoder_id: str | None,
) -> tuple[str, str | None, str | None, tuple[str, ...], str | None]:
    mode = _scope_name(lineage_mode, field_name="lineage_mode")
    if isinstance(training_subject_ids, str):
        raise TypeError("training_subject_ids must be a sequence, not a string")
    subjects = tuple(training_subject_ids)
    if any(not isinstance(value, str) or not value for value in subjects):
        raise ValueError("training_subject_ids must contain non-empty strings")
    if len(set(subjects)) != len(subjects):
        raise ValueError("training_subject_ids must contain unique values")
    parent_ids = (response_artifact_id, training_row_manifest_id, encoder_id)
    if mode == _EXPLORATORY_LINEAGE_MODE:
        if any(value is not None for value in parent_ids) or subjects:
            raise ValueError(
                "exploratory precision lineage cannot declare response parents"
            )
        return mode, None, None, (), None
    if mode not in {
        _RESPONSE_LINEAGE_MODE,
        _REPEATED_RESPONSE_LINEAGE_MODE,
        _REPEATED_CR2_RESPONSE_LINEAGE_MODE,
    }:
        raise ValueError("lineage_mode is not a supported precision lineage mode")
    if not subjects:
        raise ValueError("parented precision requires training_subject_ids")
    if subjects != tuple(sorted(subjects)):
        raise ValueError("training_subject_ids must use canonical sorted order")
    if any(value is None for value in parent_ids):
        raise ValueError("parented precision requires complete response parent IDs")
    validated_ids = tuple(
        _scope_name(cast(str, value), field_name=field_name)
        for value, field_name in zip(
            parent_ids,
            ("response_artifact_id", "training_row_manifest_id", "encoder_id"),
            strict=True,
        )
    )
    return mode, validated_ids[0], validated_ids[1], subjects, validated_ids[2]


def _lineage_payload(
    *,
    lineage_mode: str,
    response_artifact_id: str | None,
    training_row_manifest_id: str | None,
    training_subject_ids: tuple[str, ...],
    encoder_id: str | None,
) -> dict[str, object]:
    return {
        "encoder_id": encoder_id,
        "lineage_mode": lineage_mode,
        "response_artifact_id": response_artifact_id,
        "training_row_manifest_id": training_row_manifest_id,
        "training_subject_ids": list(training_subject_ids),
    }


def _finite_vector_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    if canonical.ndim != 1 or np.any(~np.isfinite(canonical)):
        raise ValueError("precision output digest requires a finite vector")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _raw_vector_digest(values: np.ndarray) -> str:
    """Hash finite and non-finite raw values with canonical float tokens."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("raw precision digest requires a vector")
    tokens: list[str] = []
    for value in raw:
        scalar = float(value)
        if math.isnan(scalar):
            tokens.append("nan")
        elif scalar == math.inf:
            tokens.append("+inf")
        elif scalar == -math.inf:
            tokens.append("-inf")
        else:
            tokens.append(scalar.hex())
    vector_id: str = stable_id(
        "raw_precision_vector",
        {"length": len(tokens), "float64_tokens": tokens},
        schema_version="2",
        digest_length=64,
    )
    return vector_id.rsplit("_", maxsplit=1)[1]


def _validated_feature_scale(
    feature_scale: np.ndarray | Sequence[float] | None,
    *,
    n_features: int,
) -> tuple[np.ndarray, str | None, str]:
    if feature_scale is None:
        return (
            _immutable_vector(np.empty(0, dtype=np.float64)),
            None,
            (_MISSING_SCALE_SOURCE),
        )
    scale = np.asarray(feature_scale, dtype=np.float64)
    if scale.shape != (n_features,):
        raise ValueError("feature_scale must align with response feature_ids")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("feature_scale must be finite and strictly positive")
    immutable = _immutable_vector(scale)
    return immutable, _finite_vector_digest(immutable), _CALLER_SCALE_SOURCE


def _validated_residual_df(
    response_artifact: FoldGeneResponseArtifact,
) -> tuple[int | None, str]:
    residual_df = response_artifact.residual_df
    if residual_df is not None:
        if (
            isinstance(residual_df, bool)
            or not isinstance(residual_df, int)
            or residual_df < 1
        ):
            raise ValueError("response residual_df must be a positive integer or None")
        return residual_df, _ARTIFACT_DF_SOURCE
    return None, _MISSING_DF_SOURCE


def _response_working_precision(
    raw_precision: np.ndarray,
    *,
    feature_scale: np.ndarray,
    residual_df: int | None,
) -> tuple[np.ndarray, str]:
    raw = np.asarray(raw_precision, dtype=np.float64)
    supported = np.isfinite(raw) & (raw > 0)
    if residual_df is None:
        return supported.astype(np.float64), _UNKNOWN_DF_WEIGHT_MODE
    if residual_df <= _LOW_DF_THRESHOLD:
        return supported.astype(np.float64), _LOW_DF_WEIGHT_MODE
    if feature_scale.size == 0:
        return supported.astype(np.float64), _MISSING_SCALE_WEIGHT_MODE
    with np.errstate(over="ignore", invalid="ignore"):
        standardized = raw * np.square(feature_scale)
    if np.any(~np.isfinite(standardized[supported])):
        raise ValueError(
            "standardized response precision must be finite for supported features"
        )
    return standardized, _STANDARDIZED_WEIGHT_MODE


@dataclass(frozen=True, slots=True, init=False)
class PrecisionTransformResult:
    """Producer-owned precision weights and complete fitted provenance."""

    values: np.ndarray
    feature_ids: tuple[str, ...]
    receiver: str
    contrast_name: str
    fold_id: str
    method: str
    lower_quantile: float
    upper_quantile: float
    lower_bound: float | None
    upper_bound: float | None
    normalization_median: float | None
    n_positive_features: int
    min_positive_features: int
    raw_precision_digest: str
    working_precision_digest: str
    transformed_precision_digest: str
    feature_scale: np.ndarray
    feature_scale_digest: str | None
    feature_scale_source: str | None
    residual_df: int | None
    residual_df_source: str | None
    low_df_threshold: int | None
    weight_mode: str
    lineage_mode: str
    response_artifact_id: str | None
    training_row_manifest_id: str | None
    training_subject_ids: tuple[str, ...]
    encoder_id: str | None
    precision_transform_id: str
    estimable: bool
    reason_code: str | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "PrecisionTransformResult is producer-owned; "
            "use winsorized_normalized_precision()"
        )

    @classmethod
    def _from_transform(
        cls,
        *,
        values: np.ndarray,
        feature_ids: tuple[str, ...],
        receiver: str,
        contrast_name: str,
        fold_id: str,
        method: str,
        lower_quantile: float,
        upper_quantile: float,
        lower_bound: float | None,
        upper_bound: float | None,
        normalization_median: float | None,
        min_positive_features: int,
        raw_precision_digest: str,
        working_precision_digest: str,
        feature_scale: np.ndarray,
        feature_scale_digest: str | None,
        feature_scale_source: str | None,
        residual_df: int | None,
        residual_df_source: str | None,
        low_df_threshold: int | None,
        weight_mode: str,
        lineage_mode: str,
        response_artifact_id: str | None,
        training_row_manifest_id: str | None,
        training_subject_ids: Sequence[str],
        encoder_id: str | None,
    ) -> PrecisionTransformResult:
        features = _names(feature_ids, field_name="feature_ids")
        receiver_name = _scope_name(receiver, field_name="receiver")
        contrast = _scope_name(contrast_name, field_name="contrast_name")
        fold = _scope_name(fold_id, field_name="fold_id")
        (
            mode,
            response_parent,
            row_manifest_parent,
            training_subjects,
            encoder_parent,
        ) = _validated_lineage(
            lineage_mode=lineage_mode,
            response_artifact_id=response_artifact_id,
            training_row_manifest_id=training_row_manifest_id,
            training_subject_ids=training_subject_ids,
            encoder_id=encoder_id,
        )
        transformed = _immutable_vector(values)
        scale = _immutable_vector(feature_scale)
        if transformed.shape != (len(features),):
            raise ContractError(
                "Precision transform values must align with feature_ids",
                code="invalid_precision_transform",
                field="values",
                remediation="Fit precision with an aligned ordered feature universe",
            )
        if np.any(~np.isfinite(transformed)) or np.any(transformed < 0):
            raise ContractError(
                "Precision transform values must be finite and non-negative",
                code="invalid_precision_transform",
                field="values",
                remediation="Build precision weights with the released transform",
            )
        n_positive = int(np.count_nonzero(transformed))
        estimable = n_positive >= min_positive_features
        reason_code = None if estimable else "insufficient_response_precision_support"
        transformed_digest = _finite_vector_digest(transformed)
        payload: dict[str, Any] = {
            "contrast_name": contrast,
            "feature_scale_digest": feature_scale_digest,
            "feature_scale_source": feature_scale_source,
            "feature_ids": list(features),
            "fold_id": fold,
            "low_df_threshold": low_df_threshold,
            "lower_bound": lower_bound,
            "lower_quantile": lower_quantile,
            "method": method,
            "min_positive_features": min_positive_features,
            "normalization_median": normalization_median,
            "raw_precision_digest": raw_precision_digest,
            "receiver": receiver_name,
            "residual_df": residual_df,
            "residual_df_source": residual_df_source,
            "transformed_precision_digest": transformed_digest,
            "upper_bound": upper_bound,
            "upper_quantile": upper_quantile,
            "weight_mode": weight_mode,
            "working_precision_digest": working_precision_digest,
        }
        payload.update(
            _lineage_payload(
                lineage_mode=mode,
                response_artifact_id=response_parent,
                training_row_manifest_id=row_manifest_parent,
                training_subject_ids=training_subjects,
                encoder_id=encoder_parent,
            )
        )
        transform_id = stable_id(
            "precision_transform", payload, schema_version=_TRANSFORM_SCHEMA_VERSION
        )
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            "values": transformed,
            "feature_ids": features,
            "receiver": receiver_name,
            "contrast_name": contrast,
            "fold_id": fold,
            "method": method,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "normalization_median": normalization_median,
            "n_positive_features": n_positive,
            "min_positive_features": min_positive_features,
            "raw_precision_digest": raw_precision_digest,
            "working_precision_digest": working_precision_digest,
            "transformed_precision_digest": transformed_digest,
            "feature_scale": scale,
            "feature_scale_digest": feature_scale_digest,
            "feature_scale_source": feature_scale_source,
            "residual_df": residual_df,
            "residual_df_source": residual_df_source,
            "low_df_threshold": low_df_threshold,
            "weight_mode": weight_mode,
            "lineage_mode": mode,
            "response_artifact_id": response_parent,
            "training_row_manifest_id": row_manifest_parent,
            "training_subject_ids": training_subjects,
            "encoder_id": encoder_parent,
            "precision_transform_id": transform_id,
            "estimable": estimable,
            "reason_code": reason_code,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)
        return self

    def _require_producer_owned(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("precision transform was not produced by this workflow")
        try:
            (
                mode,
                response_parent,
                row_manifest_parent,
                training_subjects,
                encoder_parent,
            ) = _validated_lineage(
                lineage_mode=self.lineage_mode,
                response_artifact_id=self.response_artifact_id,
                training_row_manifest_id=self.training_row_manifest_id,
                training_subject_ids=self.training_subject_ids,
                encoder_id=self.encoder_id,
            )
            transformed = np.asarray(self.values, dtype=np.float64)
            if transformed.shape != (len(self.feature_ids),):
                raise ValueError(
                    "precision transform shape does not match its feature universe"
                )
            if not _is_immutable_byte_backed(self.values):
                raise ValueError("precision transform values are not immutable")
            if not _is_immutable_byte_backed(self.feature_scale):
                raise ValueError("precision feature scale is not immutable")
            transformed_digest = _finite_vector_digest(transformed)
            scale = np.asarray(self.feature_scale, dtype=np.float64)
            if scale.ndim != 1 or (
                scale.size and (np.any(~np.isfinite(scale)) or np.any(scale <= 0))
            ):
                raise ValueError("precision feature scale is invalid")
            scale_digest = None if scale.size == 0 else _finite_vector_digest(scale)
            if self.residual_df is not None and (
                isinstance(self.residual_df, bool)
                or not isinstance(self.residual_df, int)
                or self.residual_df < 1
            ):
                raise ValueError("precision residual df is invalid")
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Precision transform failed integrity validation",
                code="precision_transform_integrity_violation",
                field="precision_transform_id",
                remediation="Refit the precision transform from training data",
            ) from error
        expected_payload: dict[str, Any] = {
            "contrast_name": self.contrast_name,
            "feature_scale_digest": self.feature_scale_digest,
            "feature_scale_source": self.feature_scale_source,
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "low_df_threshold": self.low_df_threshold,
            "lower_bound": self.lower_bound,
            "lower_quantile": self.lower_quantile,
            "method": self.method,
            "min_positive_features": self.min_positive_features,
            "normalization_median": self.normalization_median,
            "raw_precision_digest": self.raw_precision_digest,
            "receiver": self.receiver,
            "residual_df": self.residual_df,
            "residual_df_source": self.residual_df_source,
            "transformed_precision_digest": transformed_digest,
            "upper_bound": self.upper_bound,
            "upper_quantile": self.upper_quantile,
            "weight_mode": self.weight_mode,
            "working_precision_digest": self.working_precision_digest,
        }
        expected_payload.update(
            _lineage_payload(
                lineage_mode=mode,
                response_artifact_id=response_parent,
                training_row_manifest_id=row_manifest_parent,
                training_subject_ids=training_subjects,
                encoder_id=encoder_parent,
            )
        )
        expected_id = stable_id(
            "precision_transform",
            expected_payload,
            schema_version=_TRANSFORM_SCHEMA_VERSION,
        )
        exploratory_metadata_valid = (
            self.method == _EXPLORATORY_METHOD
            and self.lineage_mode == _EXPLORATORY_LINEAGE_MODE
            and scale.size == 0
            and self.feature_scale_digest is None
            and self.feature_scale_source is None
            and self.residual_df is None
            and self.residual_df_source is None
            and self.low_df_threshold is None
            and self.weight_mode == _RAW_WEIGHT_MODE
            and self.working_precision_digest == self.raw_precision_digest
        )
        response_metadata_valid = (
            self.method == _STANDARDIZED_GUARDRAIL_METHOD
            and self.lineage_mode == _RESPONSE_LINEAGE_MODE
            and self.low_df_threshold == _LOW_DF_THRESHOLD
            and self.feature_scale_source
            in {_CALLER_SCALE_SOURCE, _MISSING_SCALE_SOURCE}
            and self.residual_df_source in {_ARTIFACT_DF_SOURCE, _MISSING_DF_SOURCE}
            and self.weight_mode
            in {
                _STANDARDIZED_WEIGHT_MODE,
                _LOW_DF_WEIGHT_MODE,
                _UNKNOWN_DF_WEIGHT_MODE,
                _MISSING_SCALE_WEIGHT_MODE,
            }
            and (
                (scale.size == len(self.feature_ids))
                == (self.feature_scale_source == _CALLER_SCALE_SOURCE)
            )
            and (
                (self.residual_df is None)
                == (self.residual_df_source == _MISSING_DF_SOURCE)
            )
            and (
                self.weight_mode == _STANDARDIZED_WEIGHT_MODE
                or self.normalization_median in {None, 1.0}
            )
            and (
                self.weight_mode != _STANDARDIZED_WEIGHT_MODE
                or (
                    self.residual_df is not None
                    and self.residual_df > _LOW_DF_THRESHOLD
                    and scale.size == len(self.feature_ids)
                )
            )
            and (
                self.weight_mode != _LOW_DF_WEIGHT_MODE
                or (
                    self.residual_df is not None
                    and self.residual_df <= _LOW_DF_THRESHOLD
                )
            )
            and (
                self.weight_mode != _UNKNOWN_DF_WEIGHT_MODE or self.residual_df is None
            )
            and (
                self.weight_mode != _MISSING_SCALE_WEIGHT_MODE
                or (
                    self.residual_df is not None
                    and self.residual_df > _LOW_DF_THRESHOLD
                    and scale.size == 0
                )
            )
        )
        repeated_cr1_metadata_valid = (
            self.method == _REPEATED_STANDARDIZED_METHOD
            and self.lineage_mode == _REPEATED_RESPONSE_LINEAGE_MODE
            and self.low_df_threshold is None
            and self.feature_scale_source
            in {_CALLER_SCALE_SOURCE, _MISSING_SCALE_SOURCE}
            and self.residual_df is None
            and self.residual_df_source == _REPEATED_DF_SOURCE
            and self.weight_mode
            in {
                _REPEATED_STANDARDIZED_WEIGHT_MODE,
                _REPEATED_MISSING_SCALE_WEIGHT_MODE,
            }
            and (
                (scale.size == len(self.feature_ids))
                == (self.feature_scale_source == _CALLER_SCALE_SOURCE)
            )
            and (
                self.weight_mode == _REPEATED_STANDARDIZED_WEIGHT_MODE
                or self.normalization_median in {None, 1.0}
            )
            and (
                (self.weight_mode == _REPEATED_STANDARDIZED_WEIGHT_MODE)
                == (scale.size == len(self.feature_ids))
            )
        )
        repeated_cr2_metadata_valid = (
            self.method == _REPEATED_CR2_STANDARDIZED_METHOD
            and self.lineage_mode == _REPEATED_CR2_RESPONSE_LINEAGE_MODE
            and self.low_df_threshold is None
            and self.feature_scale_source
            in {_CALLER_SCALE_SOURCE, _MISSING_SCALE_SOURCE}
            and self.residual_df is None
            and self.residual_df_source == _REPEATED_CR2_DF_SOURCE
            and self.weight_mode
            in {
                _REPEATED_CR2_STANDARDIZED_WEIGHT_MODE,
                _REPEATED_CR2_MISSING_SCALE_WEIGHT_MODE,
            }
            and (
                (scale.size == len(self.feature_ids))
                == (self.feature_scale_source == _CALLER_SCALE_SOURCE)
            )
            and (
                self.weight_mode == _REPEATED_CR2_STANDARDIZED_WEIGHT_MODE
                or self.normalization_median in {None, 1.0}
            )
            and (
                (self.weight_mode == _REPEATED_CR2_STANDARDIZED_WEIGHT_MODE)
                == (scale.size == len(self.feature_ids))
            )
        )
        valid = (
            (
                exploratory_metadata_valid
                or response_metadata_valid
                or repeated_cr1_metadata_valid
                or repeated_cr2_metadata_valid
            )
            and scale_digest == self.feature_scale_digest
            and transformed_digest == self.transformed_precision_digest
            and int(np.count_nonzero(transformed)) == self.n_positive_features
            and self.estimable
            == (self.n_positive_features >= self.min_positive_features)
            and self.estimable == (self.reason_code is None)
            and expected_id == self.precision_transform_id
        )
        if not valid:
            raise ContractError(
                "Precision transform provenance failed integrity validation",
                code="precision_transform_integrity_violation",
                field="precision_transform_id",
                remediation="Refit the precision transform from training data",
            )

    def require_compatible(
        self,
        *,
        feature_ids: Sequence[str],
        receiver: str,
        contrast_name: str,
        fold_id: str,
    ) -> None:
        """Validate integrity and exact solver scope before consuming weights."""

        self._require_producer_owned()
        expected = (
            _names(feature_ids, field_name="feature_ids"),
            _scope_name(receiver, field_name="receiver"),
            _scope_name(contrast_name, field_name="contrast_name"),
            _scope_name(fold_id, field_name="fold_id"),
        )
        observed = (
            self.feature_ids,
            self.receiver,
            self.contrast_name,
            self.fold_id,
        )
        if observed != expected:
            raise ContractError(
                "Precision transform does not match the requested attribution scope",
                code="precision_transform_scope_mismatch",
                field="precision_transform_id",
                remediation=(
                    "Use the transform fitted for this ordered feature universe"
                ),
            )

    def require_response_compatible(
        self,
        response_artifact: (
            FoldGeneResponseArtifact | RepeatedMeasuresFoldResponseArtifact
        ),
    ) -> None:
        """Validate integrity and exact fold-response parentage."""

        if not isinstance(
            response_artifact,
            (FoldGeneResponseArtifact, RepeatedMeasuresFoldResponseArtifact),
        ):
            raise TypeError("response_artifact must be a supported fold response")
        response_artifact._require_intact()
        self._require_producer_owned()
        if isinstance(response_artifact, RepeatedMeasuresFoldResponseArtifact):
            lineage_mode = (
                _REPEATED_RESPONSE_LINEAGE_MODE
                if response_artifact.is_cr1_exploratory
                else _REPEATED_CR2_RESPONSE_LINEAGE_MODE
            )
        else:
            lineage_mode = _RESPONSE_LINEAGE_MODE
        expected = (
            lineage_mode,
            response_artifact.artifact_id,
            response_artifact.training_sample_manifest_digest,
            response_artifact.training_subject_ids,
            response_artifact.encoder_id,
            response_artifact.feature_ids,
            response_artifact.receiver,
            response_artifact.contrast_name,
            response_artifact.fold_id,
            _raw_vector_digest(response_artifact.raw_precision),
        )
        observed = (
            self.lineage_mode,
            self.response_artifact_id,
            self.training_row_manifest_id,
            self.training_subject_ids,
            self.encoder_id,
            self.feature_ids,
            self.receiver,
            self.contrast_name,
            self.fold_id,
            self.raw_precision_digest,
        )
        if observed != expected:
            raise ContractError(
                "Precision transform does not match its fold response parent",
                code="precision_transform_parent_mismatch",
                field="response_artifact_id",
                remediation="Fit precision from this exact response artifact",
            )

    def require_standardized_space_compatible(
        self,
        *,
        feature_scale: np.ndarray | Sequence[float] | None,
        residual_df: int | None,
    ) -> None:
        """Validate the exact response standardization and df guardrail inputs."""

        self._require_producer_owned()
        if self.method not in {
            _STANDARDIZED_GUARDRAIL_METHOD,
            _REPEATED_STANDARDIZED_METHOD,
            _REPEATED_CR2_STANDARDIZED_METHOD,
        }:
            raise ContractError(
                "Precision transform is not defined for a standardized response",
                code="precision_transform_standardization_mismatch",
                field="method",
                remediation="Fit response precision for this standardized space",
            )
        scale, scale_digest, _ = _validated_feature_scale(
            feature_scale,
            n_features=len(self.feature_ids),
        )
        if residual_df is not None and (
            isinstance(residual_df, (bool, np.bool_))
            or not isinstance(residual_df, (int, np.integer))
            or int(residual_df) < 1
        ):
            raise ValueError("residual_df must be a positive integer or None")
        expected_df = None if residual_df is None else int(residual_df)
        if (
            scale_digest != self.feature_scale_digest
            or scale.shape != self.feature_scale.shape
            or expected_df != self.residual_df
        ):
            raise ContractError(
                "Precision transform does not match the downstream response space",
                code="precision_transform_standardization_mismatch",
                field="feature_scale",
                remediation=(
                    "Use the transform fitted with this exact feature scale and "
                    "residual degrees of freedom"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        """Return provenance without expanding the transformed vector."""

        self._require_producer_owned()
        return {
            "precision_transform_id": self.precision_transform_id,
            "method": self.method,
            "receiver": self.receiver,
            "contrast_name": self.contrast_name,
            "fold_id": self.fold_id,
            "feature_ids": list(self.feature_ids),
            "raw_precision_digest": self.raw_precision_digest,
            "working_precision_digest": self.working_precision_digest,
            "transformed_precision_digest": self.transformed_precision_digest,
            "feature_scale_digest": self.feature_scale_digest,
            "feature_scale_source": self.feature_scale_source,
            "residual_df": self.residual_df,
            "residual_df_source": self.residual_df_source,
            "low_df_threshold": self.low_df_threshold,
            "weight_mode": self.weight_mode,
            "lineage_mode": self.lineage_mode,
            "response_artifact_id": self.response_artifact_id,
            "training_row_manifest_id": self.training_row_manifest_id,
            "training_subject_ids": list(self.training_subject_ids),
            "encoder_id": self.encoder_id,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "normalization_median": self.normalization_median,
            "n_positive_features": self.n_positive_features,
            "min_positive_features": self.min_positive_features,
            "estimable": self.estimable,
            "reason_code": self.reason_code,
        }


def _fit_precision_transform(
    raw_precision: np.ndarray,
    *,
    feature_ids: Sequence[str],
    receiver: str,
    contrast_name: str,
    fold_id: str,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_positive_features: int = 2,
    method: str,
    source_raw_precision_digest: str | None,
    feature_scale: np.ndarray,
    feature_scale_digest: str | None,
    feature_scale_source: str | None,
    residual_df: int | None,
    residual_df_source: str | None,
    low_df_threshold: int | None,
    weight_mode: str,
    lineage_mode: str,
    response_artifact_id: str | None,
    training_row_manifest_id: str | None,
    training_subject_ids: Sequence[str],
    encoder_id: str | None,
) -> PrecisionTransformResult:
    raw = np.asarray(raw_precision, dtype=np.float64).copy()
    features = _names(feature_ids, field_name="feature_ids")
    if raw.ndim != 1:
        raise ValueError("raw_precision must be one-dimensional")
    if raw.shape != (len(features),):
        raise ValueError("raw_precision must align with feature_ids")
    if isinstance(lower_quantile, (bool, np.bool_)) or isinstance(
        upper_quantile, (bool, np.bool_)
    ):
        raise ValueError("precision quantiles must be numeric, not boolean")
    try:
        lower = float(lower_quantile)
        upper = float(upper_quantile)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "precision quantiles must be finite numeric scalars"
        ) from error
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not 0 <= lower <= upper <= 1
    ):
        raise ValueError("precision quantiles must satisfy 0 <= lower <= upper <= 1")
    if (
        isinstance(min_positive_features, bool)
        or not isinstance(min_positive_features, int)
        or min_positive_features < 1
    ):
        raise ValueError("min_positive_features must be a positive integer")
    minimum_support = int(min_positive_features)

    valid = np.isfinite(raw) & (raw > 0)
    result = np.zeros(raw.shape, dtype=np.float64)
    n_valid = int(np.count_nonzero(valid))
    lower_bound: float | None = None
    upper_bound: float | None = None
    normalization_median: float | None = None
    if n_valid:
        bounds = np.asarray(
            np.quantile(raw[valid], [lower, upper]),
            dtype=np.float64,
        )
        lower_bound = float(bounds[0])
        upper_bound = float(bounds[1])
        clipped = np.clip(raw[valid], lower_bound, upper_bound)
        normalization_median = float(np.median(clipped))
        if normalization_median > 0:
            result[valid] = clipped / normalization_median

    return PrecisionTransformResult._from_transform(
        values=result,
        feature_ids=features,
        receiver=receiver,
        contrast_name=contrast_name,
        fold_id=fold_id,
        method=method,
        lower_quantile=lower,
        upper_quantile=upper,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        normalization_median=normalization_median,
        min_positive_features=minimum_support,
        raw_precision_digest=(
            _raw_vector_digest(raw)
            if source_raw_precision_digest is None
            else source_raw_precision_digest
        ),
        working_precision_digest=_raw_vector_digest(raw),
        feature_scale=feature_scale,
        feature_scale_digest=feature_scale_digest,
        feature_scale_source=feature_scale_source,
        residual_df=residual_df,
        residual_df_source=residual_df_source,
        low_df_threshold=low_df_threshold,
        weight_mode=weight_mode,
        lineage_mode=lineage_mode,
        response_artifact_id=response_artifact_id,
        training_row_manifest_id=training_row_manifest_id,
        training_subject_ids=training_subject_ids,
        encoder_id=encoder_id,
    )


def winsorized_normalized_precision(
    raw_precision: np.ndarray,
    *,
    feature_ids: Sequence[str],
    receiver: str,
    contrast_name: str,
    fold_id: str,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_positive_features: int = 2,
) -> PrecisionTransformResult:
    """Fit an exploratory transform without a response-artifact parent."""

    return _fit_precision_transform(
        raw_precision,
        feature_ids=feature_ids,
        receiver=receiver,
        contrast_name=contrast_name,
        fold_id=fold_id,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        min_positive_features=min_positive_features,
        method=_EXPLORATORY_METHOD,
        source_raw_precision_digest=None,
        feature_scale=np.empty(0, dtype=np.float64),
        feature_scale_digest=None,
        feature_scale_source=None,
        residual_df=None,
        residual_df_source=None,
        low_df_threshold=None,
        weight_mode=_RAW_WEIGHT_MODE,
        lineage_mode=_EXPLORATORY_LINEAGE_MODE,
        response_artifact_id=None,
        training_row_manifest_id=None,
        training_subject_ids=(),
        encoder_id=None,
    )


def fit_response_precision(
    response_artifact: FoldGeneResponseArtifact,
    *,
    feature_scale: np.ndarray | Sequence[float] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_positive_features: int = 2,
) -> PrecisionTransformResult:
    """Fit unit-compatible response weights with a low-df guardrail.

    ``feature_scale`` must be the positive per-feature scale used to standardize
    downstream responses. For residual df at or below four, supported features
    receive equal weight. At higher df, inverse-variance precision is converted
    to standardized-response units as ``precision * feature_scale**2`` before
    winsorization and positive-median normalization. If the downstream scale is
    unavailable, the transform fails closed to equal supported-feature weights.
    This is a deterministic guardrail, not empirical-Bayes variance moderation.
    """

    if not isinstance(response_artifact, FoldGeneResponseArtifact):
        raise TypeError("response_artifact must be a FoldGeneResponseArtifact")
    response_artifact._require_intact()
    scale, scale_digest, scale_source = _validated_feature_scale(
        feature_scale,
        n_features=len(response_artifact.feature_ids),
    )
    residual_df, residual_df_source = _validated_residual_df(response_artifact)
    working_precision, weight_mode = _response_working_precision(
        response_artifact.raw_precision,
        feature_scale=scale,
        residual_df=residual_df,
    )
    result = _fit_precision_transform(
        working_precision,
        feature_ids=response_artifact.feature_ids,
        receiver=response_artifact.receiver,
        contrast_name=response_artifact.contrast_name,
        fold_id=response_artifact.fold_id,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        min_positive_features=min_positive_features,
        method=_STANDARDIZED_GUARDRAIL_METHOD,
        source_raw_precision_digest=_raw_vector_digest(response_artifact.raw_precision),
        feature_scale=scale,
        feature_scale_digest=scale_digest,
        feature_scale_source=scale_source,
        residual_df=residual_df,
        residual_df_source=residual_df_source,
        low_df_threshold=_LOW_DF_THRESHOLD,
        weight_mode=weight_mode,
        lineage_mode=_RESPONSE_LINEAGE_MODE,
        response_artifact_id=response_artifact.artifact_id,
        training_row_manifest_id=(response_artifact.training_sample_manifest_digest),
        training_subject_ids=response_artifact.training_subject_ids,
        encoder_id=response_artifact.encoder_id,
    )
    result.require_response_compatible(response_artifact)
    result.require_standardized_space_compatible(
        feature_scale=feature_scale,
        residual_df=residual_df,
    )
    return result


def fit_repeated_response_precision(
    response_artifact: RepeatedMeasuresFoldResponseArtifact,
    *,
    feature_scale: np.ndarray | Sequence[float] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_positive_features: int = 2,
) -> PrecisionTransformResult:
    """Fit backend-specific weights with exact repeated-response parentage.

    The transform consumes CR1 diagnostic precision or strict CR2 precision as
    declared by the response parent. It only supplies relative feature weights
    and never authorizes analytic p/q values.
    """

    if not isinstance(response_artifact, RepeatedMeasuresFoldResponseArtifact):
        raise TypeError(
            "response_artifact must be a RepeatedMeasuresFoldResponseArtifact"
        )
    response_artifact._require_intact()
    is_cr1 = response_artifact.is_cr1_exploratory
    scale, scale_digest, scale_source = _validated_feature_scale(
        feature_scale,
        n_features=len(response_artifact.feature_ids),
    )
    raw = np.asarray(response_artifact.raw_precision, dtype=np.float64)
    supported = np.isfinite(raw) & (raw > 0)
    if scale.size == 0:
        working_precision = supported.astype(np.float64)
        weight_mode = _REPEATED_MISSING_SCALE_WEIGHT_MODE
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            working_precision = raw * np.square(scale)
        if np.any(~np.isfinite(working_precision[supported])):
            raise ValueError(
                "standardized repeated precision must be finite for supported features"
            )
        weight_mode = (
            _REPEATED_STANDARDIZED_WEIGHT_MODE
            if is_cr1
            else _REPEATED_CR2_STANDARDIZED_WEIGHT_MODE
        )
    if scale.size == 0:
        weight_mode = (
            _REPEATED_MISSING_SCALE_WEIGHT_MODE
            if is_cr1
            else _REPEATED_CR2_MISSING_SCALE_WEIGHT_MODE
        )
    result = _fit_precision_transform(
        working_precision,
        feature_ids=response_artifact.feature_ids,
        receiver=response_artifact.receiver,
        contrast_name=response_artifact.contrast_name,
        fold_id=response_artifact.fold_id,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        min_positive_features=min_positive_features,
        method=(
            _REPEATED_STANDARDIZED_METHOD
            if is_cr1
            else _REPEATED_CR2_STANDARDIZED_METHOD
        ),
        source_raw_precision_digest=_raw_vector_digest(response_artifact.raw_precision),
        feature_scale=scale,
        feature_scale_digest=scale_digest,
        feature_scale_source=scale_source,
        residual_df=None,
        residual_df_source=(_REPEATED_DF_SOURCE if is_cr1 else _REPEATED_CR2_DF_SOURCE),
        low_df_threshold=None,
        weight_mode=weight_mode,
        lineage_mode=(
            _REPEATED_RESPONSE_LINEAGE_MODE
            if is_cr1
            else _REPEATED_CR2_RESPONSE_LINEAGE_MODE
        ),
        response_artifact_id=response_artifact.artifact_id,
        training_row_manifest_id=(response_artifact.training_sample_manifest_digest),
        training_subject_ids=response_artifact.training_subject_ids,
        encoder_id=response_artifact.encoder_id,
    )
    result.require_response_compatible(response_artifact)
    result.require_standardized_space_compatible(
        feature_scale=feature_scale,
        residual_df=None,
    )
    return result
