"""Subject-level binary occurrence prevalence contrasts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from crychic.core import canonical_digest, stable_id

OCCURRENCE_CONTRAST_VERSION = "jeffreys_beta_prevalence_contrast_m4_v1"
OCCURRENCE_CONTRAST_COLUMNS = (
    "event_id",
    "reference",
    "target",
    "reference_subjects",
    "target_subjects",
    "reference_occurrences",
    "target_occurrences",
    "reference_posterior_prevalence",
    "target_posterior_prevalence",
    "reference_posterior_variance",
    "target_posterior_variance",
    "prevalence_difference",
    "posterior_standardized_prevalence_difference",
    "posterior_log_odds_ratio",
    "direction",
    "status",
    "reason_code",
    "input_digest",
    "spec_id",
    "occurrence_contrast_id",
    "formal_inference_allowed",
    "score_version",
)
_SCHEMA_VERSION = "1.0.0"


class OccurrenceContrastStatus(StrEnum):
    """Whether both group prevalences are descriptively estimable."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, kw_only=True)
class OccurrenceContrastSpec:
    """Fixed prior and coverage policy for independent subject groups."""

    beta_prior: float = 0.5
    minimum_subjects_per_group: int = 8
    design: str = "independent_groups"
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        prior = float(self.beta_prior)
        if not math.isfinite(prior) or prior <= 0.0:
            raise ValueError("beta_prior must be finite and positive")
        minimum = self.minimum_subjects_per_group
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
            raise ValueError("minimum_subjects_per_group must be an integer >= 2")
        if self.design != "independent_groups":
            raise ValueError("M4 v1 supports only independent_groups")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "beta_prior", prior)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "occurrence_contrast_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": OCCURRENCE_CONTRAST_VERSION,
            "beta_prior": self.beta_prior,
            "design": self.design,
            "minimum_subjects_per_group": self.minimum_subjects_per_group,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


def _canonical_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _event_digest(table: pd.DataFrame) -> str:
    records: list[dict[str, Any]] = []
    for row in table.loc[
        :, ["event_id", "subject_id", "condition", "occurrence"]
    ].itertuples(index=False):
        records.append(
            {
                "event_id": str(row.event_id),
                "subject_id": str(row.subject_id),
                "condition": str(row.condition),
                "occurrence": (
                    None if pd.isna(row.occurrence) else int(row.occurrence)
                ),
            }
        )
    return canonical_digest(records)


def _posterior_summary(
    occurrences: int, subjects: int, prior: float
) -> tuple[float, float, float]:
    alpha = occurrences + prior
    beta = subjects - occurrences + prior
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total**2 * (total + 1.0))
    log_odds = math.log(alpha) - math.log(beta)
    return mean, variance, log_odds


def _result_row(
    *,
    event_id: str,
    reference: str,
    target: str,
    reference_subjects: int,
    target_subjects: int,
    reference_occurrences: int,
    target_occurrences: int,
    spec: OccurrenceContrastSpec,
    input_digest: str,
    status: OccurrenceContrastStatus,
    reason_code: str | None,
) -> dict[str, object]:
    if status is OccurrenceContrastStatus.OBSERVED:
        reference_mean, reference_variance, reference_log_odds = _posterior_summary(
            reference_occurrences, reference_subjects, spec.beta_prior
        )
        target_mean, target_variance, target_log_odds = _posterior_summary(
            target_occurrences, target_subjects, spec.beta_prior
        )
        difference: float | None = target_mean - reference_mean
        standardized_difference: float | None = difference / math.sqrt(
            reference_variance + target_variance
        )
        log_odds_ratio: float | None = target_log_odds - reference_log_odds
        direction: int | None = int(np.sign(difference))
    else:
        reference_mean = None
        reference_variance = None
        target_mean = None
        target_variance = None
        difference = None
        standardized_difference = None
        log_odds_ratio = None
        direction = None
    identity = {
        "event_id": event_id,
        "reference": reference,
        "target": target,
        "reference_subjects": reference_subjects,
        "target_subjects": target_subjects,
        "reference_occurrences": reference_occurrences,
        "target_occurrences": target_occurrences,
        "reference_posterior_prevalence": reference_mean,
        "target_posterior_prevalence": target_mean,
        "reference_posterior_variance": reference_variance,
        "target_posterior_variance": target_variance,
        "prevalence_difference": difference,
        "posterior_standardized_prevalence_difference": standardized_difference,
        "posterior_log_odds_ratio": log_odds_ratio,
        "direction": direction,
        "status": status.value,
        "reason_code": reason_code,
        "input_digest": input_digest,
        "spec_id": spec.spec_id,
        "formal_inference_allowed": False,
        "score_version": OCCURRENCE_CONTRAST_VERSION,
    }
    return {
        **identity,
        "occurrence_contrast_id": stable_id(
            "subject_occurrence_contrast",
            identity,
            schema_version=_SCHEMA_VERSION,
        ),
    }


def fit_subject_occurrence_contrasts(
    subject_events: pd.DataFrame,
    *,
    reference: str,
    target: str,
    spec: OccurrenceContrastSpec | None = None,
) -> pd.DataFrame:
    """Estimate separate group prevalence and a descriptive contrast per event.

    Missing occurrence values remain missing and reduce observed subject
    coverage; they are never converted to absence. The v1 estimator is for
    independent groups and deliberately emits no p value, q value, or event
    activity probability.
    """

    if not isinstance(subject_events, pd.DataFrame):
        raise TypeError("subject_events must be a pandas DataFrame")
    resolved = spec or OccurrenceContrastSpec()
    if not isinstance(resolved, OccurrenceContrastSpec):
        raise TypeError("spec must be an OccurrenceContrastSpec or None")
    reference = _canonical_name(reference, field_name="reference")
    target = _canonical_name(target, field_name="target")
    if reference == target:
        raise ValueError("reference and target must differ")
    required = {"event_id", "subject_id", "condition", "occurrence"}
    missing = required.difference(subject_events.columns)
    if missing:
        raise ValueError(f"subject events are missing columns: {sorted(missing)}")
    source = subject_events.loc[:, sorted(required)].copy()
    if source.empty:
        raise ValueError("subject events must be non-empty")
    for column in ("event_id", "subject_id", "condition"):
        if source[column].isna().any():
            raise ValueError(f"{column} cannot be missing")
        source[column] = source[column].astype(str)
        invalid = source[column].eq("") | source[column].str.strip().ne(source[column])
        if invalid.any():
            raise ValueError(f"{column} values must be canonical non-empty strings")
    conditions = set(source["condition"])
    if conditions != {reference, target}:
        raise ValueError("subject events must contain exactly reference and target")
    if source.duplicated(["event_id", "subject_id", "condition"]).any():
        raise ValueError("subject events require one row per event, subject, condition")
    subjects_by_condition = source.groupby("condition", observed=True)[
        "subject_id"
    ].unique()
    if set(subjects_by_condition[reference]).intersection(
        subjects_by_condition[target]
    ):
        raise ValueError(
            "independent_groups cannot reuse subject IDs across conditions"
        )
    numeric = pd.to_numeric(source["occurrence"], errors="coerce")
    invalid_numeric = source["occurrence"].notna() & numeric.isna()
    finite = numeric.dropna()
    if invalid_numeric.any() or not finite.isin([0.0, 1.0]).all():
        raise ValueError("finite occurrence values must equal zero or one")
    source["occurrence"] = numeric.astype(float)
    source = source.sort_values(
        ["event_id", "condition", "subject_id"], kind="stable", ignore_index=True
    )

    rows: list[dict[str, object]] = []
    for event_id, event in source.groupby("event_id", observed=True, sort=True):
        observed = event.dropna(subset=["occurrence"])
        counts = observed.groupby("condition", observed=True)["occurrence"].agg(
            ["count", "sum"]
        )
        reference_subjects = (
            int(counts.loc[reference, "count"]) if reference in counts.index else 0
        )
        target_subjects = (
            int(counts.loc[target, "count"]) if target in counts.index else 0
        )
        reference_occurrences = (
            int(counts.loc[reference, "sum"]) if reference in counts.index else 0
        )
        target_occurrences = (
            int(counts.loc[target, "sum"]) if target in counts.index else 0
        )
        estimable = min(reference_subjects, target_subjects) >= (
            resolved.minimum_subjects_per_group
        )
        rows.append(
            _result_row(
                event_id=str(event_id),
                reference=reference,
                target=target,
                reference_subjects=reference_subjects,
                target_subjects=target_subjects,
                reference_occurrences=reference_occurrences,
                target_occurrences=target_occurrences,
                spec=resolved,
                input_digest=_event_digest(event),
                status=(
                    OccurrenceContrastStatus.OBSERVED
                    if estimable
                    else OccurrenceContrastStatus.NOT_ESTIMABLE
                ),
                reason_code=(
                    None if estimable else "insufficient_observed_subjects_per_group"
                ),
            )
        )
    result = pd.DataFrame.from_records(rows)
    return result.loc[:, list(OCCURRENCE_CONTRAST_COLUMNS)].sort_values(
        "event_id", kind="stable", ignore_index=True
    )


__all__ = [
    "OCCURRENCE_CONTRAST_COLUMNS",
    "OCCURRENCE_CONTRAST_VERSION",
    "OccurrenceContrastSpec",
    "OccurrenceContrastStatus",
    "fit_subject_occurrence_contrasts",
]
