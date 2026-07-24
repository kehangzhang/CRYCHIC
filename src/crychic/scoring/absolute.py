"""Experimental absolute activity heads separated from sender attribution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from crychic.availability import BatchAvailability
from crychic.sender import SenderAssignment

ABSOLUTE_ACTIVITY_SCORE_VERSION = "log_reference_additive_m0_v1"
ABSOLUTE_ACTIVITY_HEAD_COLUMNS = (
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "ligand_absolute_evidence",
    "receptor_absolute_evidence",
    "parent_activity_raw",
    "sender_detection",
    "mechanism_support",
    "sender_attribution",
    "sender_attribution_with_null",
    "null_sender_attribution",
    "sender_mass",
    "candidate_sender_count",
    "status",
    "reason_code",
    "attribution_status",
    "attribution_reason_code",
    "score_version",
    "formal_inference_allowed",
)

_EDGE_KEYS = (
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_PARENT_KEYS = (
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "interaction_id",
)
_ASSIGNMENT_KEYS = ("context_id", "sender", "receiver", "interaction_id")
_REQUIRED_AVAILABILITY = {
    *_EDGE_KEYS,
    "ligand_absolute_evidence",
    "receptor_absolute_evidence",
    "absolute_lr_activity",
    "availability_state",
    "state_status",
    "state_reason_code",
}


def _unit_series(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    invalid = values.notna() & numeric.isna()
    finite = numeric.dropna()
    if invalid.any() or (
        not finite.empty
        and ((finite < 0.0).any() or (finite > 1.0).any() or np.isinf(finite).any())
    ):
        raise ValueError(f"{field} must contain values in [0, 1] or missing")
    return numeric


def _require_unique(table: pd.DataFrame, keys: Sequence[str], *, name: str) -> None:
    if table.duplicated(list(keys)).any():
        raise ValueError(f"{name} keys must be unique")


def build_absolute_activity_heads(
    availability: BatchAvailability,
    sender_assignment: SenderAssignment | None = None,
) -> pd.DataFrame:
    """Build M0 activity, detection, and attribution as distinct quantities.

    Detection is the sender-specific absolute LR evidence and therefore does not
    conserve parent mass. The legacy softmax remains a conditional attribution.
    A data-derived null-sender mass is added without changing detection: the
    strongest absolute ligand evidence sets the active-sender mass, and the
    remainder is assigned to the null class.

    This experimental descriptive head emits no probabilities or formal
    inferential quantities.
    """

    if not isinstance(availability, BatchAvailability):
        raise TypeError("availability must be a BatchAvailability")
    source = availability.sample_interactions.copy(deep=True)
    missing = _REQUIRED_AVAILABILITY.difference(source.columns)
    if missing:
        raise ValueError(
            f"availability lacks absolute-activity columns: {sorted(missing)}"
        )
    if source.empty:
        return pd.DataFrame(columns=ABSOLUTE_ACTIVITY_HEAD_COLUMNS)
    _require_unique(source, _EDGE_KEYS, name="availability")
    for column in (
        "ligand_absolute_evidence",
        "receptor_absolute_evidence",
        "absolute_lr_activity",
        "availability_state",
    ):
        source[column] = _unit_series(source[column], field=column)

    grouped = source.groupby(list(_PARENT_KEYS), observed=True, sort=False)
    source["parent_activity_raw"] = grouped["absolute_lr_activity"].transform("max")
    source["sender_detection"] = source["absolute_lr_activity"]
    source["mechanism_support"] = source["availability_state"]
    source["candidate_sender_count"] = grouped["sender"].transform("size").astype(int)
    active_sender_mass = grouped["ligand_absolute_evidence"].transform("max")
    source["null_sender_attribution"] = 1.0 - active_sender_mass

    if sender_assignment is None:
        source["sender_attribution"] = np.nan
        source["attribution_status"] = "not_estimable"
        source["attribution_reason_code"] = "sender_assignment_not_supplied"
    else:
        if not isinstance(sender_assignment, SenderAssignment):
            raise TypeError("sender_assignment must be a SenderAssignment or None")
        assignment = sender_assignment.table.loc[
            :,
            [
                *_ASSIGNMENT_KEYS,
                "assignment_weight",
                "status",
                "reason_code",
            ],
        ].rename(
            columns={
                "status": "attribution_status",
                "reason_code": "attribution_reason_code",
            }
        )
        _require_unique(assignment, _ASSIGNMENT_KEYS, name="sender assignment")
        assignment["_assignment_candidate_count"] = assignment.groupby(
            ["context_id", "receiver", "interaction_id"],
            observed=True,
            sort=False,
        )["sender"].transform("size")
        assignment["_assignment_row_present"] = True
        source = source.merge(
            assignment,
            on=list(_ASSIGNMENT_KEYS),
            how="left",
            validate="many_to_one",
            sort=False,
        )
        source["sender_attribution"] = _unit_series(
            source.pop("assignment_weight"), field="assignment_weight"
        )
        source["attribution_status"] = source["attribution_status"].astype("object")
        source["attribution_reason_code"] = source["attribution_reason_code"].astype(
            "object"
        )
        absent = source["attribution_status"].isna()
        source.loc[absent, "attribution_status"] = "not_estimable"
        source.loc[absent, "attribution_reason_code"] = "sender_assignment_row_missing"
        attribution_groups = source.groupby(
            list(_PARENT_KEYS), observed=True, sort=False
        )
        expected_count_variants = attribution_groups[
            "_assignment_candidate_count"
        ].transform(lambda values: values.dropna().nunique())
        if expected_count_variants.gt(1).any():
            raise RuntimeError(
                "sender assignment candidate counts differ within parent"
            )
        expected_count = attribution_groups["_assignment_candidate_count"].transform(
            "max"
        )
        source["_assignment_candidate_coverage_complete"] = (
            expected_count.notna()
            & source["candidate_sender_count"].eq(expected_count)
            & source["_assignment_row_present"].fillna(False).astype(bool)
        )
        parent_complete = attribution_groups[
            "_assignment_candidate_coverage_complete"
        ].transform("all")
        incomplete = ~parent_complete
        source.loc[incomplete, "sender_attribution"] = np.nan
        source.loc[incomplete, "attribution_status"] = "not_estimable"
        source.loc[incomplete, "attribution_reason_code"] = (
            "sender_candidate_coverage_incomplete"
        )

    active_sender_mass = 1.0 - source["null_sender_attribution"]
    source["sender_attribution_with_null"] = (
        active_sender_mass * source["sender_attribution"]
    )
    source["sender_mass"] = source["parent_activity_raw"] * source["sender_attribution"]
    observed = source["sender_detection"].notna()
    source["status"] = np.where(observed, "observed", "not_estimable")
    source["reason_code"] = None
    source.loc[~observed, "reason_code"] = source.loc[
        ~observed, "state_reason_code"
    ].where(source.loc[~observed, "state_reason_code"].notna(), "activity_missing")
    source["score_version"] = ABSOLUTE_ACTIVITY_SCORE_VERSION
    source["formal_inference_allowed"] = False

    complete = source["sender_attribution"].notna()
    if complete.any():
        summaries = (
            source.loc[complete]
            .groupby(list(_PARENT_KEYS), observed=True, sort=False)
            .agg(
                candidate_count=("sender", "size"),
                expected_count=("candidate_sender_count", "first"),
                conditional_sum=("sender_attribution", "sum"),
                null_adjusted_sum=("sender_attribution_with_null", "sum"),
                null_mass=("null_sender_attribution", "first"),
                sender_mass_sum=("sender_mass", "sum"),
                parent=("parent_activity_raw", "first"),
            )
        )
        fully_observed = summaries["candidate_count"].eq(summaries["expected_count"])
        checked = summaries.loc[fully_observed]
        if not np.allclose(checked["conditional_sum"], 1.0, rtol=1e-10, atol=1e-12):
            raise RuntimeError("conditional sender attribution does not conserve one")
        if not np.allclose(
            checked["null_adjusted_sum"] + checked["null_mass"],
            1.0,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError("null-aware sender attribution does not conserve one")
        if not np.allclose(
            checked["sender_mass_sum"],
            checked["parent"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError("sender visualization mass does not conserve parent")

    result = source.loc[:, list(ABSOLUTE_ACTIVITY_HEAD_COLUMNS)].copy()
    return result.sort_values(list(_EDGE_KEYS), kind="stable", ignore_index=True)


__all__ = [
    "ABSOLUTE_ACTIVITY_HEAD_COLUMNS",
    "ABSOLUTE_ACTIVITY_SCORE_VERSION",
    "build_absolute_activity_heads",
]
