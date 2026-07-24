"""Empirical-Bayes shrinkage for fold-training residualized sender coupling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import canonical_digest, stable_id

from .coupling import (
    ResidualizedCouplingSpec,
    ResidualizedCouplingStatus,
    ResidualizedSenderCoupling,
    fit_residualized_sender_coupling,
)

EB_SHRUNKEN_COUPLING_V2_VERSION = "fisher_z_zero_mean_eb_coupling_m2_v2"
_SCHEMA_VERSION = "2.0.0"
_KEY = ("sender", "receiver", "interaction_id")
_BASE_COLUMNS = {
    "subject_id",
    "sender",
    "receiver",
    "interaction_id",
    "sender_effect",
    "receiver_effect",
}


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(sorted(_name(value, field_name=field_name) for value in values))
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must be unique")
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


class EBShrunkenCouplingStatus(StrEnum):
    """Whether the shrunken coupling prior is available."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, kw_only=True)
class EBShrunkenCouplingV2Spec:
    """Frozen nuisance, EB, and attribution-weight policy for M2."""

    numerical_covariates: tuple[str, ...] = ()
    categorical_covariates: tuple[str, ...] = ()
    minimum_subjects: int = 8
    minimum_observed_edges_for_eb: int = 4
    maximum_condition_number: float = 1.0e8
    residual_norm_tolerance: float = 1.0e-12
    correlation_clip: float = 1.0 - 1.0e-6
    attribution_coupling_weight: float = 0.25
    tau2_estimator: str = "zero_mean_second_moment_v1"
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        numerical = _names(self.numerical_covariates, field_name="numerical_covariates")
        categorical = _names(
            self.categorical_covariates, field_name="categorical_covariates"
        )
        if set(numerical).intersection(categorical):
            raise ValueError("coupling numerical and categorical covariates overlap")
        for field_name, minimum in (
            ("minimum_subjects", 4),
            ("minimum_observed_edges_for_eb", 2),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        condition = _finite(
            self.maximum_condition_number,
            field_name="maximum_condition_number",
        )
        tolerance = _finite(
            self.residual_norm_tolerance,
            field_name="residual_norm_tolerance",
        )
        correlation_clip = _finite(self.correlation_clip, field_name="correlation_clip")
        coupling_weight = _finite(
            self.attribution_coupling_weight,
            field_name="attribution_coupling_weight",
        )
        if condition <= 1.0 or tolerance <= 0.0:
            raise ValueError("coupling numerical tolerances are invalid")
        if not 0.0 < correlation_clip < 1.0:
            raise ValueError("correlation_clip must lie in (0, 1)")
        if coupling_weight not in {0.0, 0.25, 0.5}:
            raise ValueError(
                "attribution_coupling_weight must be one of 0, 0.25, or 0.5"
            )
        if self.tau2_estimator != "zero_mean_second_moment_v1":
            raise ValueError("tau2_estimator is unsupported")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "numerical_covariates", numerical)
        object.__setattr__(self, "categorical_covariates", categorical)
        object.__setattr__(self, "maximum_condition_number", condition)
        object.__setattr__(self, "residual_norm_tolerance", tolerance)
        object.__setattr__(self, "correlation_clip", correlation_clip)
        object.__setattr__(self, "attribution_coupling_weight", coupling_weight)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "eb_shrunken_coupling_v2_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "attribution_coupling_weight": self.attribution_coupling_weight,
            "categorical_covariates": list(self.categorical_covariates),
            "correlation_clip": self.correlation_clip,
            "maximum_condition_number": self.maximum_condition_number,
            "minimum_observed_edges_for_eb": self.minimum_observed_edges_for_eb,
            "minimum_subjects": self.minimum_subjects,
            "numerical_covariates": list(self.numerical_covariates),
            "residual_norm_tolerance": self.residual_norm_tolerance,
            "schema_version": self.schema_version,
            "tau2_estimator": self.tau2_estimator,
            "version": EB_SHRUNKEN_COUPLING_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}

    def residualized_spec(self) -> ResidualizedCouplingSpec:
        """Return the exact v1 nuisance fit used before EB shrinkage."""

        return ResidualizedCouplingSpec(
            numerical_covariates=self.numerical_covariates,
            categorical_covariates=self.categorical_covariates,
            minimum_subjects=self.minimum_subjects,
            maximum_condition_number=self.maximum_condition_number,
            residual_norm_tolerance=self.residual_norm_tolerance,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EBShrunkenCouplingRecordV2:
    """One candidate edge's raw and EB-shrunken training coupling."""

    sender: str
    receiver: str
    interaction_id: str
    base_coupling_id: str
    n_complete_subjects: int
    nuisance_parameter_count: int
    raw_correlation: float | None
    fisher_z: float | None
    sampling_variance: float | None
    shrinkage_factor: float | None
    shrunken_correlation: float | None
    status: EBShrunkenCouplingStatus | str
    reason_code: str | None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "sender",
            "receiver",
            "interaction_id",
            "base_coupling_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("n_complete_subjects", "nuisance_parameter_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        status = EBShrunkenCouplingStatus(self.status)
        numeric = {
            name: None if value is None else _finite(value, field_name=name)
            for name, value in (
                ("raw_correlation", self.raw_correlation),
                ("fisher_z", self.fisher_z),
                ("sampling_variance", self.sampling_variance),
                ("shrinkage_factor", self.shrinkage_factor),
                ("shrunken_correlation", self.shrunken_correlation),
            )
        }
        raw = numeric["raw_correlation"]
        fisher_z = numeric["fisher_z"]
        variance = numeric["sampling_variance"]
        factor = numeric["shrinkage_factor"]
        shrunken = numeric["shrunken_correlation"]
        if raw is not None and not -1.0 <= raw <= 1.0:
            raise ValueError("raw_correlation must lie in [-1, 1]")
        if variance is not None and variance <= 0.0:
            raise ValueError("sampling_variance must be positive")
        if status is EBShrunkenCouplingStatus.OBSERVED:
            if (
                raw is None
                or fisher_z is None
                or variance is None
                or factor is None
                or not 0.0 <= factor <= 1.0
                or shrunken is None
                or not -1.0 <= shrunken <= 1.0
                or self.reason_code is not None
            ):
                raise ValueError("observed EB coupling record is incomplete")
            if not math.isclose(
                shrunken,
                math.tanh(factor * fisher_z),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("shrunken correlation does not match Fisher-z rule")
        elif self.reason_code is None or factor is not None or shrunken is not None:
            raise ValueError("unavailable EB coupling requires a reason")
        for field_name, value in numeric.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "eb_shrunken_coupling_record_v2",
                self.to_dict(include_id=False),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "base_coupling_id": self.base_coupling_id,
            "fisher_z": self.fisher_z,
            "interaction_id": self.interaction_id,
            "n_complete_subjects": self.n_complete_subjects,
            "nuisance_parameter_count": self.nuisance_parameter_count,
            "raw_correlation": self.raw_correlation,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "sampling_variance": self.sampling_variance,
            "sender": self.sender,
            "shrinkage_factor": self.shrinkage_factor,
            "shrunken_correlation": self.shrunken_correlation,
            "status": EBShrunkenCouplingStatus(self.status).value,
        }
        return {"record_id": self.record_id, **result} if include_id else result


@dataclass(frozen=True, slots=True, kw_only=True)
class EBShrunkenCouplingFunctionalV2:
    """Fold-training collection sharing one zero-mean EB variance."""

    fold_id: str
    training_subject_ids: tuple[str, ...]
    training_input_digest: str
    sender_activity_transform_id: str
    receiver_program_functional_id: str
    input_digest: str
    tau2: float | None
    records: tuple[EBShrunkenCouplingRecordV2, ...]
    spec: EBShrunkenCouplingV2Spec
    status: EBShrunkenCouplingStatus | str
    reason_code: str | None
    formal_inference_allowed: bool = False
    functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "fold_id",
            "training_input_digest",
            "sender_activity_transform_id",
            "receiver_program_functional_id",
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
        if not isinstance(self.spec, EBShrunkenCouplingV2Spec):
            raise TypeError("spec must be EBShrunkenCouplingV2Spec")
        supplied_records = tuple(self.records)
        if not supplied_records or any(
            not isinstance(item, EBShrunkenCouplingRecordV2)
            for item in supplied_records
        ):
            raise TypeError("records must contain EB coupling records")
        records = tuple(
            sorted(
                supplied_records,
                key=lambda item: (item.sender, item.receiver, item.interaction_id),
            )
        )
        keys = [(item.sender, item.receiver, item.interaction_id) for item in records]
        if len(keys) != len(set(keys)):
            raise ValueError("EB coupling record keys must be unique")
        status = EBShrunkenCouplingStatus(self.status)
        tau2 = None if self.tau2 is None else _finite(self.tau2, field_name="tau2")
        if status is EBShrunkenCouplingStatus.OBSERVED:
            if tau2 is None or tau2 < 0.0 or self.reason_code is not None:
                raise ValueError("observed EB functional requires non-negative tau2")
            if not any(
                item.status is EBShrunkenCouplingStatus.OBSERVED for item in records
            ):
                raise ValueError("observed EB functional requires observed records")
        elif tau2 is not None or self.reason_code is None:
            raise ValueError("unavailable EB functional requires a reason")
        if self.formal_inference_allowed is not False:
            raise ValueError("coupling prior cannot claim formal inference")
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "tau2", tau2)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "functional_id",
            stable_id(
                "eb_shrunken_coupling_functional_v2",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "formal_inference_allowed": self.formal_inference_allowed,
            "input_digest": self.input_digest,
            "reason_code": self.reason_code,
            "receiver_program_functional_id": self.receiver_program_functional_id,
            "records": [item.to_dict() for item in self.records],
            "sender_activity_transform_id": self.sender_activity_transform_id,
            "spec_id": self.spec.spec_id,
            "status": EBShrunkenCouplingStatus(self.status).value,
            "tau2": self.tau2,
            "training_input_digest": self.training_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
            "version": EB_SHRUNKEN_COUPLING_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_id": self.functional_id,
            **self._identity_payload(),
            "spec": self.spec.to_dict(),
        }


def _input_digest(table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    ordered = table.loc[:, list(columns)].sort_values(
        [*_KEY, "subject_id"], kind="stable"
    )
    records: list[list[Any]] = []
    for row in ordered.itertuples(index=False, name=None):
        record: list[Any] = []
        for value in row:
            if pd.isna(value):
                record.append(None)
            elif isinstance(value, (float, np.floating)):
                record.append({"float_hex": float(value).hex()})
            elif isinstance(value, np.generic):
                record.append(value.item())
            else:
                record.append(value)
        records.append(record)
    return cast(str, canonical_digest(records))


def _raw_fits(
    table: pd.DataFrame,
    *,
    fold_id: str,
    spec: EBShrunkenCouplingV2Spec,
) -> tuple[ResidualizedSenderCoupling, ...]:
    residual_spec = spec.residualized_spec()
    fits = []
    keep_columns = (
        "subject_id",
        "sender_effect",
        "receiver_effect",
        *spec.numerical_covariates,
        *spec.categorical_covariates,
    )
    for key, group in table.groupby(list(_KEY), observed=True, sort=True):
        sender, receiver, interaction_id = (str(value) for value in key)
        fits.append(
            fit_residualized_sender_coupling(
                group.loc[:, list(keep_columns)],
                sender=sender,
                receiver=receiver,
                interaction_id=interaction_id,
                fold_id=fold_id,
                spec=residual_spec,
            )
        )
    return tuple(fits)


def fit_eb_shrunken_coupling_v2(
    subject_effects: pd.DataFrame,
    *,
    fold_id: str,
    training_subject_ids: tuple[str, ...],
    training_input_digest: str,
    sender_activity_transform_id: str,
    receiver_program_functional_id: str,
    spec: EBShrunkenCouplingV2Spec | None = None,
) -> EBShrunkenCouplingFunctionalV2:
    """Fit residual correlations and shrink Fisher-z values across candidate edges."""

    resolved = spec or EBShrunkenCouplingV2Spec()
    if not isinstance(resolved, EBShrunkenCouplingV2Spec):
        raise TypeError("spec must be EBShrunkenCouplingV2Spec or None")
    fold = _name(fold_id, field_name="fold_id")
    subjects = _names(training_subject_ids, field_name="training_subject_ids")
    required = {
        *_BASE_COLUMNS,
        *resolved.numerical_covariates,
        *resolved.categorical_covariates,
    }
    if not isinstance(subject_effects, pd.DataFrame):
        raise TypeError("subject_effects must be a pandas DataFrame")
    missing = required.difference(subject_effects.columns)
    if missing:
        raise ValueError(f"coupling v2 input is missing columns: {sorted(missing)}")
    table = subject_effects.loc[:, sorted(required)].copy(deep=True)
    for column in ("subject_id", *_KEY):
        if table[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        table[column] = table[column].astype(str)
        if (
            table[column].eq("").any()
            or table[column].str.strip().ne(table[column]).any()
        ):
            raise ValueError(f"{column} must contain canonical identifiers")
    if table.duplicated([*_KEY, "subject_id"]).any():
        raise ValueError("coupling v2 requires one row per edge and subject")
    observed_subjects = tuple(sorted(table["subject_id"].unique()))
    if observed_subjects != subjects:
        raise ValueError("coupling rows must exactly cover training_subject_ids")
    digest_columns = (
        "subject_id",
        *_KEY,
        "sender_effect",
        "receiver_effect",
        *resolved.numerical_covariates,
        *resolved.categorical_covariates,
    )
    input_digest = _input_digest(table, digest_columns)
    raw_fits = _raw_fits(table, fold_id=fold, spec=resolved)

    eligible: list[tuple[ResidualizedSenderCoupling, float, float]] = []
    for fitted in raw_fits:
        if (
            fitted.status is not ResidualizedCouplingStatus.OBSERVED
            or fitted.signed_correlation is None
        ):
            continue
        nuisance_count = max(0, fitted.design_rank - 1)
        denominator = fitted.n_complete_subjects - nuisance_count - 3
        if denominator <= 0:
            continue
        clipped = float(
            np.clip(
                fitted.signed_correlation,
                -resolved.correlation_clip,
                resolved.correlation_clip,
            )
        )
        eligible.append((fitted, math.atanh(clipped), 1.0 / denominator))
    eb_estimable = len(eligible) >= resolved.minimum_observed_edges_for_eb
    tau2 = (
        max(0.0, float(np.mean([z * z - variance for _, z, variance in eligible])))
        if eb_estimable
        else None
    )
    eligible_by_id = {
        fitted.coupling_id: (z_value, variance)
        for fitted, z_value, variance in eligible
    }
    records: list[EBShrunkenCouplingRecordV2] = []
    for fitted in raw_fits:
        values = eligible_by_id.get(fitted.coupling_id)
        nuisance_count = max(0, fitted.design_rank - 1)
        if values is None:
            raw = fitted.signed_correlation
            fisher_z = variance = None
            factor = shrunken = None
            reason = fitted.reason_code or "invalid_fisher_sampling_variance"
            status = EBShrunkenCouplingStatus.NOT_ESTIMABLE
        else:
            fisher_z, variance = values
            raw = fitted.signed_correlation
            if tau2 is None:
                factor = shrunken = None
                reason = "insufficient_observed_edges_for_eb"
                status = EBShrunkenCouplingStatus.NOT_ESTIMABLE
            else:
                factor = tau2 / (tau2 + variance)
                shrunken = math.tanh(factor * fisher_z)
                reason = None
                status = EBShrunkenCouplingStatus.OBSERVED
        records.append(
            EBShrunkenCouplingRecordV2(
                sender=fitted.sender,
                receiver=fitted.receiver,
                interaction_id=fitted.interaction_id,
                base_coupling_id=fitted.coupling_id,
                n_complete_subjects=fitted.n_complete_subjects,
                nuisance_parameter_count=nuisance_count,
                raw_correlation=raw,
                fisher_z=fisher_z,
                sampling_variance=variance,
                shrinkage_factor=factor,
                shrunken_correlation=shrunken,
                status=status,
                reason_code=reason,
            )
        )
    return EBShrunkenCouplingFunctionalV2(
        fold_id=fold,
        training_subject_ids=subjects,
        training_input_digest=_name(
            training_input_digest, field_name="training_input_digest"
        ),
        sender_activity_transform_id=_name(
            sender_activity_transform_id,
            field_name="sender_activity_transform_id",
        ),
        receiver_program_functional_id=_name(
            receiver_program_functional_id,
            field_name="receiver_program_functional_id",
        ),
        input_digest=input_digest,
        tau2=tau2,
        records=tuple(records),
        spec=resolved,
        status=(
            EBShrunkenCouplingStatus.OBSERVED
            if eb_estimable
            else EBShrunkenCouplingStatus.NOT_ESTIMABLE
        ),
        reason_code=(None if eb_estimable else "insufficient_observed_edges_for_eb"),
    )


__all__ = [
    "EB_SHRUNKEN_COUPLING_V2_VERSION",
    "EBShrunkenCouplingFunctionalV2",
    "EBShrunkenCouplingRecordV2",
    "EBShrunkenCouplingStatus",
    "EBShrunkenCouplingV2Spec",
    "fit_eb_shrunken_coupling_v2",
]
