"""Fold-training-only residualized sender-receiver coupling evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from crychic.core import canonical_digest, stable_id

RESIDUALIZED_COUPLING_VERSION = "subject_contrast_residual_correlation_m2_v1"
_SCHEMA_VERSION = "1.0.0"
_REQUIRED_COLUMNS = ("subject_id", "sender_effect", "receiver_effect")


class ResidualizedCouplingStatus(StrEnum):
    """Whether a fold-training coupling coefficient is estimable."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique non-empty names")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidualizedCouplingSpec:
    """Pre-registered nuisance and numerical policy for one coupling fit."""

    numerical_covariates: tuple[str, ...] = ()
    categorical_covariates: tuple[str, ...] = ()
    minimum_subjects: int = 8
    maximum_condition_number: float = 1.0e8
    residual_norm_tolerance: float = 1.0e-12
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        numerical = _names(self.numerical_covariates, field_name="numerical_covariates")
        categorical = _names(
            self.categorical_covariates, field_name="categorical_covariates"
        )
        overlap = set(numerical).intersection(categorical)
        if overlap:
            raise ValueError(f"coupling covariates overlap: {sorted(overlap)}")
        minimum = self.minimum_subjects
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 4:
            raise ValueError("minimum_subjects must be an integer >= 4")
        maximum = float(self.maximum_condition_number)
        tolerance = float(self.residual_norm_tolerance)
        if not math.isfinite(maximum) or maximum <= 1.0:
            raise ValueError("maximum_condition_number must be finite and > 1")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_norm_tolerance must be finite and positive")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "numerical_covariates", numerical)
        object.__setattr__(self, "categorical_covariates", categorical)
        object.__setattr__(self, "maximum_condition_number", maximum)
        object.__setattr__(self, "residual_norm_tolerance", tolerance)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "residualized_coupling_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": RESIDUALIZED_COUPLING_VERSION,
            "categorical_covariates": list(self.categorical_covariates),
            "maximum_condition_number": self.maximum_condition_number,
            "minimum_subjects": self.minimum_subjects,
            "numerical_covariates": list(self.numerical_covariates),
            "residual_norm_tolerance": self.residual_norm_tolerance,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidualizedSenderCoupling:
    """One immutable descriptive coupling fitted only on training subjects."""

    sender: str
    receiver: str
    interaction_id: str
    fold_id: str
    training_subject_ids: tuple[str, ...]
    complete_subject_ids: tuple[str, ...]
    spec: ResidualizedCouplingSpec
    input_digest: str
    design_columns: tuple[str, ...]
    n_input_subjects: int
    n_complete_subjects: int
    design_rank: int
    design_condition_number: float | None
    signed_correlation: float | None
    positive_coupling_support: float | None
    status: ResidualizedCouplingStatus | str
    reason_code: str | None
    formal_inference_allowed: bool = False
    coupling_id: str = field(init=False)
    algorithm_version: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = (self.sender, self.receiver, self.interaction_id, self.fold_id)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in identifiers
        ):
            raise ValueError("coupling identifiers must be canonical strings")
        training = _names(self.training_subject_ids, field_name="training_subject_ids")
        complete = _names(self.complete_subject_ids, field_name="complete_subject_ids")
        if tuple(sorted(training)) != training or tuple(sorted(complete)) != complete:
            raise ValueError("coupling subject IDs must use sorted canonical order")
        if not set(complete).issubset(training):
            raise ValueError("complete subjects must be a training subset")
        if not isinstance(self.spec, ResidualizedCouplingSpec):
            raise TypeError("spec must be a ResidualizedCouplingSpec")
        status = ResidualizedCouplingStatus(self.status)
        counts = (
            self.n_input_subjects,
            self.n_complete_subjects,
            self.design_rank,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("coupling counts must be non-negative integers")
        if self.n_input_subjects != len(training) or self.n_complete_subjects != len(
            complete
        ):
            raise ValueError("coupling counts disagree with subject lineage")
        condition = self.design_condition_number
        if condition is not None and (not math.isfinite(condition) or condition < 1.0):
            raise ValueError("design_condition_number is invalid")
        correlation = self.signed_correlation
        support = self.positive_coupling_support
        if status is ResidualizedCouplingStatus.OBSERVED:
            if self.reason_code is not None or correlation is None or support is None:
                raise ValueError("observed coupling requires values and no reason")
            if not -1.0 <= correlation <= 1.0 or not 0.0 <= support <= 1.0:
                raise ValueError("coupling values are outside their declared scales")
            if not math.isclose(support, max(correlation, 0.0), abs_tol=1e-12):
                raise ValueError("positive coupling support disagrees with correlation")
        elif self.reason_code is None or correlation is not None or support is not None:
            raise ValueError(
                "not-estimable coupling requires a reason and missing values"
            )
        if self.formal_inference_allowed is not False:
            raise ValueError(
                "residualized coupling is descriptive, not formal inference"
            )
        if len(self.input_digest) != 64:
            raise ValueError("input_digest must be a SHA-256-compatible digest")
        object.__setattr__(self, "training_subject_ids", training)
        object.__setattr__(self, "complete_subject_ids", complete)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "algorithm_version", RESIDUALIZED_COUPLING_VERSION)
        object.__setattr__(
            self,
            "coupling_id",
            stable_id(
                "residualized_sender_coupling",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": RESIDUALIZED_COUPLING_VERSION,
            "complete_subject_ids": list(self.complete_subject_ids),
            "design_columns": list(self.design_columns),
            "design_condition_number": self.design_condition_number,
            "design_rank": self.design_rank,
            "fold_id": self.fold_id,
            "formal_inference_allowed": self.formal_inference_allowed,
            "input_digest": self.input_digest,
            "interaction_id": self.interaction_id,
            "n_complete_subjects": self.n_complete_subjects,
            "n_input_subjects": self.n_input_subjects,
            "positive_coupling_support": self.positive_coupling_support,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "sender": self.sender,
            "signed_correlation": self.signed_correlation,
            "spec_id": self.spec.spec_id,
            "status": ResidualizedCouplingStatus(self.status).value,
            "training_subject_ids": list(self.training_subject_ids),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "coupling_id": self.coupling_id,
            **self._identity_payload(),
            "spec": self.spec.to_dict(),
            "excluded_output_kinds": [
                "p_value",
                "q_value",
                "posterior_probability",
                "communication_probability",
            ],
        }


def _input_digest(table: pd.DataFrame, columns: Sequence[str]) -> str:
    records: list[dict[str, Any]] = []
    for row in table.loc[:, list(columns)].itertuples(index=False, name=None):
        records.append(
            {
                column: (
                    None
                    if pd.isna(value)
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for column, value in zip(columns, row, strict=True)
            }
        )
    return canonical_digest(records)


def _not_estimable(
    *,
    sender: str,
    receiver: str,
    interaction_id: str,
    fold_id: str,
    training_subject_ids: tuple[str, ...],
    complete_subject_ids: tuple[str, ...],
    spec: ResidualizedCouplingSpec,
    input_digest: str,
    design_columns: tuple[str, ...],
    design_rank: int,
    reason_code: str,
    condition_number: float | None = None,
) -> ResidualizedSenderCoupling:
    return ResidualizedSenderCoupling(
        sender=sender,
        receiver=receiver,
        interaction_id=interaction_id,
        fold_id=fold_id,
        training_subject_ids=training_subject_ids,
        complete_subject_ids=complete_subject_ids,
        spec=spec,
        input_digest=input_digest,
        design_columns=design_columns,
        n_input_subjects=len(training_subject_ids),
        n_complete_subjects=len(complete_subject_ids),
        design_rank=design_rank,
        design_condition_number=condition_number,
        signed_correlation=None,
        positive_coupling_support=None,
        status=ResidualizedCouplingStatus.NOT_ESTIMABLE,
        reason_code=reason_code,
    )


def _design_matrix(
    table: pd.DataFrame,
    spec: ResidualizedCouplingSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    parts: list[np.ndarray] = [np.ones((len(table), 1), dtype=float)]
    columns: list[str] = ["intercept"]
    for name in spec.numerical_covariates:
        values = table[name].to_numpy(dtype=float)
        centered = values - float(values.mean())
        scale = float(np.linalg.norm(centered))
        if scale <= spec.residual_norm_tolerance:
            continue
        parts.append((centered / scale)[:, np.newaxis])
        columns.append(f"numeric:{name}")
    for name in spec.categorical_covariates:
        levels = tuple(sorted(table[name].astype(str).unique()))
        for level in levels[1:]:
            parts.append(
                table[name].astype(str).eq(level).to_numpy(dtype=float)[:, np.newaxis]
            )
            columns.append(f"categorical:{name}={level}")
    return np.hstack(parts), tuple(columns)


def fit_residualized_sender_coupling(
    subject_effects: pd.DataFrame,
    *,
    sender: str,
    receiver: str,
    interaction_id: str,
    fold_id: str,
    spec: ResidualizedCouplingSpec | None = None,
) -> ResidualizedSenderCoupling:
    """Fit signed residual correlation from one fold's training subjects.

    Input rows are already subject-level contrast effects. The function does
    not inspect condition labels, cell-level values, truth labels, or held-out
    subjects. Both sender and receiver effects are residualized on the same
    pre-registered nuisance design before correlation.
    """

    if not isinstance(subject_effects, pd.DataFrame):
        raise TypeError("subject_effects must be a pandas DataFrame")
    resolved = spec or ResidualizedCouplingSpec()
    if not isinstance(resolved, ResidualizedCouplingSpec):
        raise TypeError("spec must be a ResidualizedCouplingSpec or None")
    required = {
        *_REQUIRED_COLUMNS,
        *resolved.numerical_covariates,
        *resolved.categorical_covariates,
    }
    missing = required.difference(subject_effects.columns)
    if missing:
        raise ValueError(f"subject effects are missing columns: {sorted(missing)}")
    source = subject_effects.loc[:, sorted(required)].copy()
    if source.empty or source["subject_id"].isna().any():
        raise ValueError("subject effects require non-empty subject IDs")
    source["subject_id"] = source["subject_id"].astype(str)
    if (
        source["subject_id"].str.strip().ne(source["subject_id"]).any()
        or source["subject_id"].eq("").any()
    ):
        raise ValueError("subject IDs must be canonical non-empty strings")
    if source["subject_id"].duplicated().any():
        raise ValueError("subject effects must contain one row per subject")
    source = source.sort_values("subject_id", kind="stable", ignore_index=True)
    training_subject_ids = tuple(source["subject_id"])
    digest_columns = (
        "subject_id",
        "sender_effect",
        "receiver_effect",
        *resolved.numerical_covariates,
        *resolved.categorical_covariates,
    )
    input_digest = _input_digest(source, digest_columns)

    numeric_columns = (
        "sender_effect",
        "receiver_effect",
        *resolved.numerical_covariates,
    )
    numeric = source.loc[:, list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = source.loc[:, list(numeric_columns)].notna() & numeric.isna()
    if invalid_numeric.any().any():
        raise ValueError("coupling numeric columns contain non-numeric values")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    categorical_complete = (
        source.loc[:, list(resolved.categorical_covariates)].notna().all(axis=1)
        if resolved.categorical_covariates
        else pd.Series(True, index=source.index)
    )
    complete_mask = (
        numeric.notna().all(axis=1).to_numpy()
        & finite
        & categorical_complete.to_numpy()
    )
    complete = source.loc[complete_mask].copy()
    complete.loc[:, list(numeric_columns)] = numeric.loc[complete_mask].to_numpy()
    complete_subject_ids = tuple(complete["subject_id"])
    if len(complete) < resolved.minimum_subjects:
        return _not_estimable(
            sender=sender,
            receiver=receiver,
            interaction_id=interaction_id,
            fold_id=fold_id,
            training_subject_ids=training_subject_ids,
            complete_subject_ids=complete_subject_ids,
            spec=resolved,
            input_digest=input_digest,
            design_columns=(),
            design_rank=0,
            reason_code="insufficient_complete_training_subjects",
        )

    design, design_columns = _design_matrix(complete, resolved)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        return _not_estimable(
            sender=sender,
            receiver=receiver,
            interaction_id=interaction_id,
            fold_id=fold_id,
            training_subject_ids=training_subject_ids,
            complete_subject_ids=complete_subject_ids,
            spec=resolved,
            input_digest=input_digest,
            design_columns=design_columns,
            design_rank=rank,
            reason_code="rank_deficient_nuisance_design",
        )
    condition = float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition > resolved.maximum_condition_number:
        return _not_estimable(
            sender=sender,
            receiver=receiver,
            interaction_id=interaction_id,
            fold_id=fold_id,
            training_subject_ids=training_subject_ids,
            complete_subject_ids=complete_subject_ids,
            spec=resolved,
            input_digest=input_digest,
            design_columns=design_columns,
            design_rank=rank,
            reason_code="ill_conditioned_nuisance_design",
            condition_number=condition,
        )
    if len(complete) < rank + 3:
        return _not_estimable(
            sender=sender,
            receiver=receiver,
            interaction_id=interaction_id,
            fold_id=fold_id,
            training_subject_ids=training_subject_ids,
            complete_subject_ids=complete_subject_ids,
            spec=resolved,
            input_digest=input_digest,
            design_columns=design_columns,
            design_rank=rank,
            reason_code="insufficient_residual_degrees_of_freedom",
            condition_number=condition,
        )
    sender_values = complete["sender_effect"].to_numpy(dtype=float)
    receiver_values = complete["receiver_effect"].to_numpy(dtype=float)
    sender_residual = (
        sender_values - design @ np.linalg.lstsq(design, sender_values, rcond=None)[0]
    )
    receiver_residual = (
        receiver_values
        - design @ np.linalg.lstsq(design, receiver_values, rcond=None)[0]
    )
    sender_norm = float(np.linalg.norm(sender_residual))
    receiver_norm = float(np.linalg.norm(receiver_residual))
    if (
        sender_norm <= resolved.residual_norm_tolerance
        or receiver_norm <= resolved.residual_norm_tolerance
    ):
        return _not_estimable(
            sender=sender,
            receiver=receiver,
            interaction_id=interaction_id,
            fold_id=fold_id,
            training_subject_ids=training_subject_ids,
            complete_subject_ids=complete_subject_ids,
            spec=resolved,
            input_digest=input_digest,
            design_columns=design_columns,
            design_rank=rank,
            reason_code="residual_variance_zero",
            condition_number=condition,
        )
    correlation = float(
        np.clip(
            np.dot(sender_residual, receiver_residual) / (sender_norm * receiver_norm),
            -1.0,
            1.0,
        )
    )
    return ResidualizedSenderCoupling(
        sender=sender,
        receiver=receiver,
        interaction_id=interaction_id,
        fold_id=fold_id,
        training_subject_ids=training_subject_ids,
        complete_subject_ids=complete_subject_ids,
        spec=resolved,
        input_digest=input_digest,
        design_columns=design_columns,
        n_input_subjects=len(training_subject_ids),
        n_complete_subjects=len(complete_subject_ids),
        design_rank=rank,
        design_condition_number=condition,
        signed_correlation=correlation,
        positive_coupling_support=max(correlation, 0.0),
        status=ResidualizedCouplingStatus.OBSERVED,
        reason_code=None,
    )


__all__ = [
    "RESIDUALIZED_COUPLING_VERSION",
    "ResidualizedCouplingSpec",
    "ResidualizedCouplingStatus",
    "ResidualizedSenderCoupling",
    "fit_residualized_sender_coupling",
]
