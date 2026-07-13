"""Fold-frozen precision transforms for receiver attribution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from crychic.core import ContractError, stable_id

_METHOD = "winsorized_median_normalized_v2"
_PRODUCER_MARKER = "crychic.attribution.precision.v2"


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


@dataclass(frozen=True, slots=True, init=False)
class PrecisionTransformResult:
    """Producer-owned winsorized weights and complete fitted provenance."""

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
    transformed_precision_digest: str
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
        lower_quantile: float,
        upper_quantile: float,
        lower_bound: float | None,
        upper_bound: float | None,
        normalization_median: float | None,
        min_positive_features: int,
        raw_precision_digest: str,
    ) -> PrecisionTransformResult:
        features = _names(feature_ids, field_name="feature_ids")
        receiver_name = _scope_name(receiver, field_name="receiver")
        contrast = _scope_name(contrast_name, field_name="contrast_name")
        fold = _scope_name(fold_id, field_name="fold_id")
        transformed = _immutable_vector(values)
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
            "feature_ids": list(features),
            "fold_id": fold,
            "lower_bound": lower_bound,
            "lower_quantile": lower_quantile,
            "method": _METHOD,
            "min_positive_features": min_positive_features,
            "normalization_median": normalization_median,
            "raw_precision_digest": raw_precision_digest,
            "receiver": receiver_name,
            "transformed_precision_digest": transformed_digest,
            "upper_bound": upper_bound,
            "upper_quantile": upper_quantile,
        }
        transform_id = stable_id("precision_transform", payload, schema_version="2")
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            "values": transformed,
            "feature_ids": features,
            "receiver": receiver_name,
            "contrast_name": contrast,
            "fold_id": fold,
            "method": _METHOD,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "normalization_median": normalization_median,
            "n_positive_features": n_positive,
            "min_positive_features": min_positive_features,
            "raw_precision_digest": raw_precision_digest,
            "transformed_precision_digest": transformed_digest,
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
        transformed = np.asarray(self.values, dtype=np.float64)
        if transformed.shape != (len(self.feature_ids),):
            raise ContractError(
                "Precision transform shape no longer matches its feature universe",
                code="precision_transform_integrity_violation",
                field="values",
                remediation="Refit the precision transform from training data",
            )
        try:
            transformed_digest = _finite_vector_digest(transformed)
        except ValueError as error:
            raise ContractError(
                "Precision transform values failed integrity validation",
                code="precision_transform_integrity_violation",
                field="values",
                remediation="Refit the precision transform from training data",
            ) from error
        expected_payload: dict[str, Any] = {
            "contrast_name": self.contrast_name,
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "lower_bound": self.lower_bound,
            "lower_quantile": self.lower_quantile,
            "method": self.method,
            "min_positive_features": self.min_positive_features,
            "normalization_median": self.normalization_median,
            "raw_precision_digest": self.raw_precision_digest,
            "receiver": self.receiver,
            "transformed_precision_digest": transformed_digest,
            "upper_bound": self.upper_bound,
            "upper_quantile": self.upper_quantile,
        }
        expected_id = stable_id(
            "precision_transform", expected_payload, schema_version="2"
        )
        valid = (
            self.method == _METHOD
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
            "transformed_precision_digest": self.transformed_precision_digest,
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
    """Winsorize precision while binding every fit input to one fold scope."""

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
        lower_quantile=lower,
        upper_quantile=upper,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        normalization_median=normalization_median,
        min_positive_features=minimum_support,
        raw_precision_digest=_raw_vector_digest(raw),
    )
