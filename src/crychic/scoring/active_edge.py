"""Frozen candidate universe for G3-P active-edge calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import pandas as pd

from crychic.core import CommunicationMode, ContractError, stable_id

_SCHEMA_VERSION = "1.0.0"
_VIEW = "gene"
_SCORE_DIRECTION = "larger_is_more_active"
_STRATUM_SEMANTICS = "score_version_gene_view_mode_contrast_context_receiver_v1"
_POINT_SEMANTICS = "technical_sample_mean_then_subject_equal_context_edge_mean_v1"
ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID: str = stable_id(
    "active_edge_candidate_universe_policy",
    {
        "candidate_grain": ("contrast_context_mode_receiver_sender_interaction_v1"),
        "complete_candidate_coverage": True,
        "point_semantics": _POINT_SEMANTICS,
        "score_direction": _SCORE_DIRECTION,
        "stratum_semantics": _STRATUM_SEMANTICS,
        "view": _VIEW,
    },
    schema_version="1",
)
_POINT_COLUMNS = (
    "sample_id",
    "subject_id",
    "repeat_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "mode",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_version",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or bool(pd.isna(cast(Any, value)))


def _canonical_scalar(value: object) -> object:
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        value = cast(Any, value).item()
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _point_source_digest(table: pd.DataFrame) -> str:
    rows = [
        [_canonical_scalar(value) for value in row]
        for row in table.loc[:, _POINT_COLUMNS].itertuples(index=False, name=None)
    ]
    rows.sort(key=repr)
    identifier: str = stable_id(
        "active_edge_point_source_table",
        {"columns": list(_POINT_COLUMNS), "rows": rows},
        schema_version=_SCHEMA_VERSION,
        digest_length=64,
    )
    return identifier


class ActiveEdgePointStatus(StrEnum):
    """Availability of one subject-equal observed active-edge statistic."""

    OBSERVED = "observed"
    STRUCTURAL_ZERO = "structural_zero"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveEdgeCandidate:
    """One preregistered context sender-LR-receiver probability candidate."""

    contrast_id: str
    context_id: str
    sender: str
    receiver: str
    interaction_id: str
    driver_id: str
    mode: CommunicationMode | str
    candidate_edge_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            name: _name(getattr(self, name), field_name=name)
            for name in (
                "contrast_id",
                "context_id",
                "sender",
                "receiver",
                "interaction_id",
                "driver_id",
            )
        }

        mode = CommunicationMode(self.mode)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "candidate_edge_id",
            stable_id(
                "active_edge_candidate",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def biological_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.context_id,
            self.sender,
            self.receiver,
            self.interaction_id,
            CommunicationMode(self.mode).value,
        )

    def stratum_id(self, *, score_version: str) -> str:
        version = _name(score_version, field_name="score_version")
        identifier: str = stable_id(
            "active_probability_stratum",
            {
                "score_version": version,
                "view": _VIEW,
                "mode": CommunicationMode(self.mode).value,
                "contrast_id": self.contrast_id,
                "context_id": self.context_id,
                "receiver": self.receiver,
                "stratum_semantics": _STRATUM_SEMANTICS,
            },
            schema_version=_SCHEMA_VERSION,
        )
        return identifier

    def _identity_payload(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "context_id": self.context_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "interaction_id": self.interaction_id,
            "driver_id": self.driver_id,
            "mode": CommunicationMode(self.mode).value,
            "view": _VIEW,
            "score_direction": _SCORE_DIRECTION,
        }

    def _require_intact(self) -> None:
        try:
            repeated = ActiveEdgeCandidate(
                contrast_id=self.contrast_id,
                context_id=self.context_id,
                sender=self.sender,
                receiver=self.receiver,
                interaction_id=self.interaction_id,
                driver_id=self.driver_id,
                mode=self.mode,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.candidate_edge_id == repeated.candidate_edge_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Active-edge candidate failed integrity validation",
                code="active_edge_candidate_integrity_violation",
                field="candidate_edge_id",
                remediation="Refreeze the candidate from canonical edge fields",
            ) from error
        if not valid:
            raise ContractError(
                "Active-edge candidate failed integrity validation",
                code="active_edge_candidate_integrity_violation",
                field="candidate_edge_id",
                remediation="Refreeze the candidate from canonical edge fields",
            )

    def to_dict(self, *, score_version: str) -> dict[str, object]:
        self._require_intact()
        return {
            "candidate_edge_id": self.candidate_edge_id,
            **self._identity_payload(),
            "stratum_id": self.stratum_id(score_version=score_version),
        }


@dataclass(frozen=True, slots=True, init=False)
class SubjectEqualActiveEdgeScore:
    """Producer-owned point statistic for one frozen active-edge candidate."""

    candidate_edge_id: str
    stratum_id: str
    source_collection_id: str
    source_table_digest: str
    score_version: str
    score: float | None
    status: ActiveEdgePointStatus
    reason_code: str | None
    n_source_rows: int
    n_subjects: int
    point_score_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "SubjectEqualActiveEdgeScore is producer-owned; use "
            "summarize_subject_equal_active_edge_score()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "candidate_edge_id": self.candidate_edge_id,
            "stratum_id": self.stratum_id,
            "source_collection_id": self.source_collection_id,
            "source_table_digest": self.source_table_digest,
            "score_version": self.score_version,
            "score": self.score,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_source_rows": self.n_source_rows,
            "n_subjects": self.n_subjects,
            "point_semantics": _POINT_SEMANTICS,
            "score_direction": _SCORE_DIRECTION,
            "producer_marker": "crychic.scoring.subject_equal_active_edge_score.v1",
        }

    def _require_intact(self) -> None:
        try:
            expected_id = stable_id(
                "subject_equal_active_edge_score",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker
                == "crychic.scoring.subject_equal_active_edge_score.v1"
                and self.point_score_id == expected_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Subject-equal active-edge score failed integrity validation",
                code="active_edge_point_score_integrity_violation",
                field="point_score_id",
                remediation="Reaggregate the exact intact OOF score rows",
            ) from error
        if not valid:
            raise ContractError(
                "Subject-equal active-edge score failed integrity validation",
                code="active_edge_point_score_integrity_violation",
                field="point_score_id",
                remediation="Reaggregate the exact intact OOF score rows",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {"point_score_id": self.point_score_id, **payload}


@dataclass(frozen=True, slots=True, init=False)
class FrozenActiveEdgeUniverse:
    """Producer-owned complete candidate set for one score collection."""

    universe_name: str
    contrast_id: str
    score_version: str
    candidates: tuple[ActiveEdgeCandidate, ...]
    candidate_edge_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    universe_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenActiveEdgeUniverse is producer-owned; use "
            "freeze_active_edge_universe()"
        )

    @property
    def candidate_universe_policy_id(self) -> str:
        """Return the portable policy identity used by G3-P calibration."""

        return ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_name": self.universe_name,
            "contrast_id": self.contrast_id,
            "score_version": self.score_version,
            "candidate_edge_ids": list(self.candidate_edge_ids),
            "stratum_ids": list(self.stratum_ids),
            "view": _VIEW,
            "score_direction": _SCORE_DIRECTION,
            "stratum_semantics": _STRATUM_SEMANTICS,
            "complete_candidate_coverage": True,
            "candidate_universe_policy_id": self.candidate_universe_policy_id,
            "producer_marker": "crychic.scoring.frozen_active_edge_universe.v1",
        }

    def _require_intact(self) -> None:
        try:
            for candidate in self.candidates:
                candidate._require_intact()
            repeated = freeze_active_edge_universe(
                self.candidates,
                universe_name=self.universe_name,
                contrast_id=self.contrast_id,
                score_version=self.score_version,
            )
            valid = (
                self._producer_marker
                == "crychic.scoring.frozen_active_edge_universe.v1"
                and self.candidate_edge_ids == repeated.candidate_edge_ids
                and self.stratum_ids == repeated.stratum_ids
                and self._identity_payload() == repeated._identity_payload()
                and self.universe_id == repeated.universe_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen active-edge universe failed integrity validation",
                code="active_edge_universe_integrity_violation",
                field="universe_id",
                remediation="Refreeze the complete preregistered edge universe",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen active-edge universe failed integrity validation",
                code="active_edge_universe_integrity_violation",
                field="universe_id",
                remediation="Refreeze the complete preregistered edge universe",
            )

    def candidate_for(self, candidate_edge_id: str) -> ActiveEdgeCandidate:
        self._require_intact()
        edge_id = _name(candidate_edge_id, field_name="candidate_edge_id")
        matched = tuple(
            candidate
            for candidate in self.candidates
            if candidate.candidate_edge_id == edge_id
        )
        if len(matched) != 1:
            raise KeyError(edge_id)
        return matched[0]

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "universe_id": self.universe_id,
            **payload,
            "candidates": [
                candidate.to_dict(score_version=self.score_version)
                for candidate in self.candidates
            ],
        }


def summarize_subject_equal_active_edge_score(
    candidate: ActiveEdgeCandidate,
    rows: pd.DataFrame,
    *,
    source_collection_id: str,
    score_version: str,
) -> SubjectEqualActiveEdgeScore:
    """Reduce exact OOF sample rows without weighting subjects by sample count."""

    if not isinstance(candidate, ActiveEdgeCandidate):
        raise TypeError("candidate must be an ActiveEdgeCandidate")
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    candidate._require_intact()
    collection_id = _name(
        source_collection_id,
        field_name="source_collection_id",
    )
    version = _name(score_version, field_name="score_version")
    missing = set(_POINT_COLUMNS).difference(rows.columns)
    if missing:
        raise ValueError(
            f"active-edge point rows are missing columns: {sorted(missing)}"
        )
    frame = rows.loc[:, list(_POINT_COLUMNS)].copy(deep=True)
    source_digest = _point_source_digest(frame)
    n_rows = len(frame)
    n_subjects = int(frame["subject_id"].nunique()) if n_rows else 0
    score: float | None
    status: ActiveEdgePointStatus
    reason: str | None
    if frame.empty:
        score = None
        status = ActiveEdgePointStatus.NOT_ESTIMABLE
        reason = "active_edge_candidate_missing_from_oof_scores"
    else:
        identity_fields = {
            "context_id": candidate.context_id,
            "sender": candidate.sender,
            "receiver": candidate.receiver,
            "interaction_id": candidate.interaction_id,
            "mode": CommunicationMode(candidate.mode).value,
            "score_version": version,
        }
        for field_name, expected in identity_fields.items():
            observed = set(frame[field_name].astype(str))
            if observed != {expected}:
                raise ContractError(
                    "OOF point rows do not match the frozen active-edge candidate",
                    code="active_edge_point_source_mismatch",
                    field=field_name,
                    remediation=(
                        "Select rows from the exact contrast collection and candidate"
                    ),
                )
        for field_name in ("sample_id", "subject_id", "repeat_id"):
            missing_identifier = frame[field_name].isna().any()
            empty_identifier = frame[field_name].astype(str).eq("").any()
            if missing_identifier or empty_identifier:
                raise ValueError(f"{field_name} must be complete and non-empty")
        sample_subject = frame.loc[:, ["sample_id", "subject_id"]].drop_duplicates()
        if sample_subject["sample_id"].duplicated().any():
            raise ContractError(
                "One OOF sample is assigned to multiple subjects",
                code="active_edge_point_sample_subject_mismatch",
                field="sample_id,subject_id",
                remediation="Use the intact sample-subject lineage",
            )
        if frame.duplicated(["repeat_id", "sample_id", "subject_id"]).any():
            raise ContractError(
                "Active-edge point source contains duplicate repeat-sample rows",
                code="duplicate_active_edge_point_source_row",
                field="repeat_id,sample_id,subject_id,candidate_edge_id",
                remediation="Retain each held-out sample exactly once within a repeat",
            )
        normalized_scores: list[float | None] = []
        normalized_statuses: list[ActiveEdgePointStatus] = []
        for source in frame.itertuples(index=False):
            raw_status = str(source.status)
            try:
                row_status = ActiveEdgePointStatus(raw_status)
            except ValueError as error:
                raise ValueError(
                    f"unsupported active-edge point status {raw_status!r}"
                ) from error
            raw_score = source.global_sender_lr_score
            raw_reason = source.reason_code
            if row_status is ActiveEdgePointStatus.OBSERVED:
                if _is_missing(raw_score) or not _is_missing(raw_reason):
                    raise ValueError("observed point rows require score and no reason")
                numeric = float(cast(Any, raw_score))
                if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                    raise ValueError("active-edge point score must lie in [0, 1]")
            elif row_status is ActiveEdgePointStatus.STRUCTURAL_ZERO:
                if _is_missing(raw_score) or float(cast(Any, raw_score)) != 0.0:
                    raise ValueError("structural-zero point rows require exact zero")
                _name(raw_reason, field_name="reason_code")
                numeric = 0.0
            else:
                if not _is_missing(raw_score):
                    raise ValueError("not-estimable point rows require a missing score")
                _name(raw_reason, field_name="reason_code")
                numeric = math.nan
            normalized_scores.append(
                None if row_status is ActiveEdgePointStatus.NOT_ESTIMABLE else numeric
            )
            normalized_statuses.append(row_status)
        if ActiveEdgePointStatus.NOT_ESTIMABLE in normalized_statuses:
            score = None
            status = ActiveEdgePointStatus.NOT_ESTIMABLE
            reason = "active_edge_point_source_not_estimable"
        else:
            frame["_numeric_score"] = cast(list[float], normalized_scores)
            repeat_scores = frame.groupby(
                ["subject_id", "repeat_id"],
                sort=True,
                observed=True,
            )["_numeric_score"].mean()
            subject_scores = repeat_scores.groupby(
                "subject_id",
                sort=True,
                observed=True,
            ).mean()
            score = float(subject_scores.mean())
            if score == 0.0:
                status = ActiveEdgePointStatus.STRUCTURAL_ZERO
                reason = "active_edge_point_all_subject_scores_structural_zero"
            else:
                status = ActiveEdgePointStatus.OBSERVED
                reason = None
    self = object.__new__(SubjectEqualActiveEdgeScore)
    values: dict[str, object] = {
        "candidate_edge_id": candidate.candidate_edge_id,
        "stratum_id": candidate.stratum_id(score_version=version),
        "source_collection_id": collection_id,
        "source_table_digest": source_digest,
        "score_version": version,
        "score": score,
        "status": status,
        "reason_code": reason,
        "n_source_rows": n_rows,
        "n_subjects": n_subjects,
        "_producer_marker": "crychic.scoring.subject_equal_active_edge_score.v1",
    }
    for field_name, value in values.items():
        object.__setattr__(self, field_name, value)
    object.__setattr__(
        self,
        "point_score_id",
        stable_id(
            "subject_equal_active_edge_score",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


def freeze_active_edge_universe(
    candidates: tuple[ActiveEdgeCandidate, ...],
    *,
    universe_name: str,
    contrast_id: str,
    score_version: str,
) -> FrozenActiveEdgeUniverse:
    """Freeze one exact contrast-bound active-edge candidate universe."""

    name = _name(universe_name, field_name="universe_name")
    contrast = _name(contrast_id, field_name="contrast_id")
    version = _name(score_version, field_name="score_version")
    supplied = tuple(candidates)
    if not supplied or any(
        not isinstance(candidate, ActiveEdgeCandidate) for candidate in supplied
    ):
        raise ValueError("candidates must contain ActiveEdgeCandidate values")
    for candidate in supplied:
        candidate._require_intact()
    ordered = tuple(sorted(supplied, key=lambda item: item.candidate_edge_id))
    if any(candidate.contrast_id != contrast for candidate in ordered):
        raise ContractError(
            "Active-edge candidates use a different contrast collection",
            code="active_edge_universe_contrast_mismatch",
            field="contrast_id",
            remediation="Freeze each contrast-common collection separately",
        )
    ids = tuple(candidate.candidate_edge_id for candidate in ordered)
    biological_keys = tuple(candidate.biological_key for candidate in ordered)
    if len(ids) != len(set(ids)) or len(biological_keys) != len(set(biological_keys)):
        raise ContractError(
            "Active-edge candidate universe contains duplicate biological keys",
            code="duplicate_active_edge_candidate",
            field="candidate_edge_id,biological_key",
            remediation="Declare every context sender-LR-receiver edge once",
        )
    strata = tuple(candidate.stratum_id(score_version=version) for candidate in ordered)
    self = object.__new__(FrozenActiveEdgeUniverse)
    values: dict[str, object] = {
        "universe_name": name,
        "contrast_id": contrast,
        "score_version": version,
        "candidates": ordered,
        "candidate_edge_ids": ids,
        "stratum_ids": strata,
        "_producer_marker": "crychic.scoring.frozen_active_edge_universe.v1",
    }
    for field_name, value in values.items():
        object.__setattr__(self, field_name, value)
    object.__setattr__(
        self,
        "universe_id",
        stable_id(
            "frozen_active_edge_universe",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


__all__ = [
    "ACTIVE_EDGE_CANDIDATE_UNIVERSE_POLICY_ID",
    "ActiveEdgeCandidate",
    "ActiveEdgePointStatus",
    "FrozenActiveEdgeUniverse",
    "SubjectEqualActiveEdgeScore",
    "freeze_active_edge_universe",
    "summarize_subject_equal_active_edge_score",
]
