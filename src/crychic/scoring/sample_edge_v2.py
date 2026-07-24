"""Versioned sample-level edge score contract for the v7 estimator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from crychic.core import canonical_json, stable_id

SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION = "2.0.0"
SAMPLE_EDGE_SCORE_V2_COLUMNS = (
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "fold_id",
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "ligand_activity_raw",
    "receptor_activity_raw",
    "sender_detection_raw",
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
    "program_signed",
    "program_status",
    "program_reason_code",
    "program_functional_id",
    "mechanism_support",
    "coupling_prior",
    "coupling_status",
    "coupling_reason_code",
    "coupling_functional_id",
    "active_probability",
    "occurrence_status",
    "occurrence_reason_code",
    "occurrence_functional_id",
    "sender_attribution",
    "null_sender_attribution",
    "attribution_entropy",
    "attribution_status",
    "attribution_reason_code",
    "attribution_functional_id",
    "cell_count_reliability",
    "reliability_weight",
    "candidate_sender_count",
    "effective_candidate_count",
    "coverage_status",
    "structural_impossibility",
    "selection_stability",
    "status",
    "reason_code",
    "score_version",
    "activity_functional_id",
    "out_of_fold",
    "provenance",
)

_KEY = (
    "sample_id",
    "context_id",
    "fold_id",
    "sender",
    "receiver",
    "interaction_id",
)
_PARENT_KEY = (
    "sample_id",
    "context_id",
    "fold_id",
    "receiver",
    "interaction_id",
)
_REQUIRED_IDENTIFIERS = (
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "fold_id",
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "score_version",
    "activity_functional_id",
    "provenance",
)
_RAW_COLUMNS = (
    "ligand_activity_raw",
    "receptor_activity_raw",
    "sender_detection_raw",
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
)
_UNIT_INTERVAL_COLUMNS = (
    "mechanism_support",
    "active_probability",
    "sender_attribution",
    "null_sender_attribution",
    "attribution_entropy",
    "cell_count_reliability",
    "reliability_weight",
    "selection_stability",
)


class SampleEdgeCoverageStatus(StrEnum):
    """Measurement coverage without conflating absence and weak evidence."""

    MEASURED = "measured"
    LOW_COVERAGE = "low_coverage"
    STRUCTURAL_IMPOSSIBLE = "structural_impossible"
    NOT_ESTIMABLE = "not_measured_or_not_estimable"


class SampleEdgeValueStatus(StrEnum):
    """Validity of the sender-specific raw activity measurement."""

    OBSERVED = "observed"
    LOW_EVIDENCE = "low_evidence"
    STRUCTURAL_IMPOSSIBLE = "structural_impossible"
    NOT_ESTIMABLE = "not_estimable"


class SampleEdgeHeadStatus(StrEnum):
    """Availability of an optional v7 score head."""

    OBSERVED = "observed"
    PARTIAL = "partial"
    NOT_ESTIMABLE = "not_estimable"
    NOT_COMPUTED = "not_computed"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(_name(value, field_name=field_name) for value in values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleEdgeScoreV2Provenance:
    """Fold-frozen lineage for one outcome-agnostic activity functional."""

    fold_id: str
    repeat_id: str
    training_subject_ids: tuple[str, ...]
    application_subject_ids: tuple[str, ...]
    training_input_digest: str
    application_input_digest: str
    config_digest: str
    resource_id: str
    resource_version: str
    resource_manifest_digest: str
    interaction_universe_id: str
    transform_manifest_id: str
    seed_lineage_id: str
    score_version: str
    package_version: str
    outcome_agnostic: bool = True
    condition_gate_used: bool = False
    schema_version: str = SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION
    provenance_id: str = field(init=False)

    def __post_init__(self) -> None:
        scalar_fields = (
            "fold_id",
            "repeat_id",
            "training_input_digest",
            "application_input_digest",
            "config_digest",
            "resource_id",
            "resource_version",
            "resource_manifest_digest",
            "interaction_universe_id",
            "transform_manifest_id",
            "seed_lineage_id",
            "score_version",
            "package_version",
        )
        for field_name in scalar_fields:
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        training = _names(
            tuple(self.training_subject_ids), field_name="training_subject_ids"
        )
        application = _names(
            tuple(self.application_subject_ids), field_name="application_subject_ids"
        )
        if set(training).intersection(application):
            raise ValueError("training and application subjects must be disjoint")
        if self.outcome_agnostic is not True:
            raise ValueError("v7 primary activity provenance must be outcome_agnostic")
        if self.condition_gate_used is not False:
            raise ValueError("v7 primary activity cannot use a condition-derived gate")
        if self.schema_version != SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "training_subject_ids", training)
        object.__setattr__(self, "application_subject_ids", application)
        object.__setattr__(
            self,
            "provenance_id",
            stable_id(
                "sample_edge_score_v2_provenance",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "application_input_digest": self.application_input_digest,
            "application_subject_ids": list(self.application_subject_ids),
            "condition_gate_used": self.condition_gate_used,
            "config_digest": self.config_digest,
            "fold_id": self.fold_id,
            "interaction_universe_id": self.interaction_universe_id,
            "outcome_agnostic": self.outcome_agnostic,
            "package_version": self.package_version,
            "repeat_id": self.repeat_id,
            "resource_id": self.resource_id,
            "resource_manifest_digest": self.resource_manifest_digest,
            "resource_version": self.resource_version,
            "schema_version": self.schema_version,
            "score_version": self.score_version,
            "seed_lineage_id": self.seed_lineage_id,
            "training_input_digest": self.training_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
            "transform_manifest_id": self.transform_manifest_id,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a canonical persistence representation."""

        return {"provenance_id": self.provenance_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SampleEdgeScoreV2Provenance:
        """Restore and verify persisted fold lineage."""

        expected = {
            "provenance_id",
            "application_input_digest",
            "application_subject_ids",
            "condition_gate_used",
            "config_digest",
            "fold_id",
            "interaction_universe_id",
            "outcome_agnostic",
            "package_version",
            "repeat_id",
            "resource_id",
            "resource_manifest_digest",
            "resource_version",
            "schema_version",
            "score_version",
            "seed_lineage_id",
            "training_input_digest",
            "training_subject_ids",
            "transform_manifest_id",
        }
        if set(value) != expected:
            raise ValueError("sample-edge provenance fields are invalid")
        result = cls(
            fold_id=value["fold_id"],
            repeat_id=value["repeat_id"],
            training_subject_ids=tuple(value["training_subject_ids"]),
            application_subject_ids=tuple(value["application_subject_ids"]),
            training_input_digest=value["training_input_digest"],
            application_input_digest=value["application_input_digest"],
            config_digest=value["config_digest"],
            resource_id=value["resource_id"],
            resource_version=value["resource_version"],
            resource_manifest_digest=value["resource_manifest_digest"],
            interaction_universe_id=value["interaction_universe_id"],
            transform_manifest_id=value["transform_manifest_id"],
            seed_lineage_id=value["seed_lineage_id"],
            score_version=value["score_version"],
            package_version=value["package_version"],
            outcome_agnostic=value["outcome_agnostic"],
            condition_gate_used=value["condition_gate_used"],
            schema_version=value["schema_version"],
        )
        if result.provenance_id != value["provenance_id"]:
            raise ValueError("sample-edge provenance ID does not match its payload")
        return result


def _numeric(
    table: pd.DataFrame,
    column: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce").astype(float)
    invalid = table[column].notna() & values.isna()
    finite = values.dropna()
    if invalid.any() or np.isinf(finite).any():
        raise ValueError(f"{column} must contain finite numeric values or NA")
    if minimum is not None and (finite < minimum).any():
        raise ValueError(f"{column} values must be >= {minimum}")
    if maximum is not None and (finite > maximum).any():
        raise ValueError(f"{column} values must be <= {maximum}")
    return values


def _same_nullable(group: pd.DataFrame, column: str) -> bool:
    values = group[column]
    if values.isna().all():
        return True
    return not values.isna().any() and values.nunique(dropna=False) == 1


@dataclass(frozen=True, slots=True)
class SampleEdgeScoreV2:
    """Validated sample x sender x LR x receiver score table."""

    table: pd.DataFrame
    provenance: SampleEdgeScoreV2Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")
        if not isinstance(self.provenance, SampleEdgeScoreV2Provenance):
            raise TypeError("provenance must be SampleEdgeScoreV2Provenance")
        if tuple(self.table.columns) != SAMPLE_EDGE_SCORE_V2_COLUMNS:
            raise ValueError("sample-edge columns do not match the v2 schema")
        table = self.table.copy(deep=True)
        if table.empty:
            object.__setattr__(self, "table", table)
            return
        for column in _REQUIRED_IDENTIFIERS:
            if table[column].isna().any() or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in table[column].tolist()
            ):
                raise ValueError(f"{column} must contain canonical identifiers")
        if table.duplicated(list(_KEY)).any():
            raise ValueError("sample-edge primary keys must be unique")
        for column in _RAW_COLUMNS:
            table[column] = _numeric(table, column, minimum=0.0)
        table["program_signed"] = _numeric(table, "program_signed")
        table["coupling_prior"] = _numeric(
            table, "coupling_prior", minimum=-1.0, maximum=1.0
        )
        for column in _UNIT_INTERVAL_COLUMNS:
            table[column] = _numeric(table, column, minimum=0.0, maximum=1.0)
        for column in ("candidate_sender_count", "effective_candidate_count"):
            numeric = _numeric(table, column, minimum=0.0)
            if not numeric.dropna().map(float.is_integer).all() or numeric.isna().any():
                raise ValueError(f"{column} must contain non-null integers")
            table[column] = numeric.astype(int)
        if (table["candidate_sender_count"] < 1).any() or (
            table["effective_candidate_count"] > table["candidate_sender_count"]
        ).any():
            raise ValueError("candidate sender counts are inconsistent")
        if any(type(value) not in {bool, np.bool_} for value in table["out_of_fold"]):
            raise ValueError("out_of_fold must contain booleans")
        if any(
            type(value) not in {bool, np.bool_}
            for value in table["structural_impossibility"]
        ):
            raise ValueError("structural_impossibility must contain booleans")
        table["out_of_fold"] = table["out_of_fold"].astype(bool)
        table["structural_impossibility"] = table["structural_impossibility"].astype(
            bool
        )
        if not table["out_of_fold"].all():
            raise ValueError("v2 primary sample-edge rows must be out of fold")

        self._validate_lineage(table)
        self._validate_statuses(table)
        self._validate_parent_geometry(table)
        self._validate_optional_heads(table)
        object.__setattr__(self, "table", table)

    def _validate_lineage(self, table: pd.DataFrame) -> None:
        expected = {
            "fold_id": self.provenance.fold_id,
            "score_version": self.provenance.score_version,
            "activity_functional_id": self.provenance.provenance_id,
            "provenance": canonical_json(self.provenance.to_dict()),
        }
        for column, value in expected.items():
            if set(table[column]) != {value}:
                raise ValueError(f"sample-edge {column} does not match provenance")
        unknown = set(table["subject_id"]).difference(
            self.provenance.application_subject_ids
        )
        if unknown:
            raise ValueError("sample-edge rows contain subjects outside held-out scope")
        sample_scope = table.groupby("sample_id", observed=True, sort=False).agg(
            subject_count=("subject_id", "nunique"),
            context_count=("context_id", "nunique"),
            condition_count=("condition", "nunique"),
            fold_count=("fold_id", "nunique"),
        )
        if (sample_scope != 1).any(axis=None):
            raise ValueError(
                "each sample must map to one subject, condition, context, fold"
            )

    def _validate_statuses(self, table: pd.DataFrame) -> None:
        allowed_values = {item.value for item in SampleEdgeValueStatus}
        allowed_coverage = {item.value for item in SampleEdgeCoverageStatus}
        if not set(table["status"]).issubset(allowed_values):
            raise ValueError("sample-edge status is unsupported")
        if not set(table["coverage_status"]).issubset(allowed_coverage):
            raise ValueError("sample-edge coverage_status is unsupported")
        status = table["status"]
        observed = status.isin(
            [
                SampleEdgeValueStatus.OBSERVED.value,
                SampleEdgeValueStatus.LOW_EVIDENCE.value,
            ]
        )
        structural = status.eq(SampleEdgeValueStatus.STRUCTURAL_IMPOSSIBLE.value)
        not_estimable = status.eq(SampleEdgeValueStatus.NOT_ESTIMABLE.value)
        if table.loc[observed, "sender_detection_raw"].isna().any():
            raise ValueError("observed or low-evidence rows require sender detection")
        if table.loc[not_estimable, "sender_detection_raw"].notna().any():
            raise ValueError("not-estimable rows require sender detection NA")
        if not np.allclose(
            table.loc[structural, "sender_detection_raw"].to_numpy(dtype=float),
            0.0,
        ):
            raise ValueError("structural-impossible detection must be zero")
        if (
            not table.loc[structural, "structural_impossibility"].all()
            or table.loc[~structural, "structural_impossibility"].any()
        ):
            raise ValueError("structural_impossibility must match status")
        if (
            not table.loc[structural, "coverage_status"]
            .eq(SampleEdgeCoverageStatus.STRUCTURAL_IMPOSSIBLE.value)
            .all()
        ):
            raise ValueError("structural rows require structural coverage status")
        if (
            not table.loc[not_estimable, "coverage_status"]
            .isin(
                [
                    SampleEdgeCoverageStatus.NOT_ESTIMABLE.value,
                    SampleEdgeCoverageStatus.LOW_COVERAGE.value,
                ]
            )
            .all()
        ):
            raise ValueError("not-estimable rows require missing or low coverage")
        status_is_observed = status.eq(SampleEdgeValueStatus.OBSERVED.value)
        if table.loc[status_is_observed, "reason_code"].notna().any():
            raise ValueError("observed rows cannot carry a reason_code")
        if table.loc[~status_is_observed, "reason_code"].isna().any():
            raise ValueError("non-observed rows require a reason_code")

    def _validate_parent_geometry(self, table: pd.DataFrame) -> None:
        for _, group in table.groupby(list(_PARENT_KEY), observed=True, sort=False):
            row_count = len(group)
            if set(group["candidate_sender_count"]) != {row_count}:
                raise ValueError("candidate_sender_count must equal parent row count")
            finite = group["sender_detection_raw"].dropna().astype(float)
            if set(group["effective_candidate_count"]) != {len(finite)}:
                raise ValueError(
                    "effective_candidate_count must equal measured senders"
                )
            for column in ("parent_peak_raw", "parent_total_raw", "parent_mean_raw"):
                if not _same_nullable(group, column):
                    raise ValueError(f"{column} must be constant within each parent")
            summaries = group.iloc[0][
                ["parent_peak_raw", "parent_total_raw", "parent_mean_raw"]
            ]
            if finite.empty:
                if summaries.notna().any():
                    raise ValueError("unmeasured parent summaries must be NA")
                continue
            expected = (float(finite.max()), float(finite.sum()), float(finite.mean()))
            observed = tuple(float(value) for value in summaries)
            if not all(
                math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(observed, expected, strict=True)
            ):
                raise ValueError("parent summaries do not match sender detections")

    def _validate_optional_heads(self, table: pd.DataFrame) -> None:
        allowed = {item.value for item in SampleEdgeHeadStatus}
        head_specs = (
            (
                "program_status",
                "program_signed",
                "program_reason_code",
                "program_functional_id",
            ),
            (
                "coupling_status",
                "coupling_prior",
                "coupling_reason_code",
                "coupling_functional_id",
            ),
            (
                "occurrence_status",
                "active_probability",
                "occurrence_reason_code",
                "occurrence_functional_id",
            ),
            (
                "attribution_status",
                "sender_attribution",
                "attribution_reason_code",
                "attribution_functional_id",
            ),
        )
        for status_column, value_column, reason_column, functional_column in head_specs:
            if not set(table[status_column]).issubset(allowed):
                raise ValueError(f"{status_column} is unsupported")
            observed = table[status_column].isin(
                [
                    SampleEdgeHeadStatus.OBSERVED.value,
                    SampleEdgeHeadStatus.PARTIAL.value,
                ]
            )
            if table.loc[observed, value_column].isna().any():
                raise ValueError(f"observed {status_column} requires {value_column}")
            if table.loc[~observed, value_column].notna().any():
                raise ValueError(
                    f"unavailable {status_column} requires {value_column}=NA"
                )
            if table.loc[observed, reason_column].notna().any():
                raise ValueError(f"observed {status_column} cannot carry a reason")
            if table.loc[~observed, reason_column].isna().any():
                raise ValueError(f"unavailable {status_column} requires a reason")
            functional_ids = table[functional_column]
            not_computed = table[status_column].eq(
                SampleEdgeHeadStatus.NOT_COMPUTED.value
            )
            if functional_ids.loc[not_computed].notna().any():
                raise ValueError(
                    f"not-computed {status_column} requires {functional_column}=NA"
                )
            if functional_ids.loc[~not_computed].isna().any():
                raise ValueError(
                    f"applied {status_column} requires {functional_column}"
                )
            supplied_ids = functional_ids.dropna()
            if any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in supplied_ids
            ):
                raise ValueError(f"{functional_column} must contain canonical IDs")
        for _, group in table.groupby(list(_PARENT_KEY), observed=True, sort=False):
            for column in (
                "program_signed",
                "program_status",
                "program_reason_code",
                "active_probability",
                "occurrence_status",
                "occurrence_reason_code",
                "null_sender_attribution",
                "attribution_entropy",
            ):
                if not _same_nullable(group, column):
                    raise ValueError(f"{column} must be constant within each parent")
            sender_weights = group["sender_attribution"].dropna().astype(float)
            null_values = group["null_sender_attribution"].dropna().astype(float)
            entropy_values = group["attribution_entropy"].dropna().astype(float)
            if sender_weights.empty:
                if not null_values.empty or not entropy_values.empty:
                    raise ValueError(
                        "null attribution and entropy require sender attribution"
                    )
                continue
            if null_values.empty:
                raise ValueError("sender attribution requires a null sender")
            total = float(sender_weights.sum()) + float(null_values.iloc[0])
            if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("sender and null attribution must sum to one")
            if entropy_values.empty:
                raise ValueError("sender attribution requires attribution entropy")
            probabilities = np.concatenate(
                ([float(null_values.iloc[0])], sender_weights.to_numpy(dtype=float))
            )
            positive = probabilities[probabilities > 0.0]
            expected_entropy = (
                -float(np.sum(positive * np.log(positive)))
                / math.log(len(probabilities))
                if len(probabilities) > 1
                else 0.0
            )
            if not math.isclose(
                float(entropy_values.iloc[0]),
                expected_entropy,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("attribution_entropy disagrees with sender weights")


__all__ = [
    "SAMPLE_EDGE_SCORE_V2_COLUMNS",
    "SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION",
    "SampleEdgeCoverageStatus",
    "SampleEdgeHeadStatus",
    "SampleEdgeScoreV2",
    "SampleEdgeScoreV2Provenance",
    "SampleEdgeValueStatus",
]
