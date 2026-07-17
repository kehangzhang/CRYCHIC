"""Strict point-workflow adapter for ADR-013 active-edge records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from crychic.core import CommunicationMode, ContractError, canonical_json, stable_id
from crychic.inference import ActiveEdgeScoreStatus, PointActiveEdgeScoreRecord
from crychic.scoring import (
    GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    ActiveEdgeCandidate,
    ActiveEdgePointStatus,
    CrossReceiverCommonScoringApplication,
    CrossReceiverCommonScoringFunctional,
    FrozenActiveEdgeUniverse,
    SubjectEqualActiveEdgeScore,
    freeze_active_edge_universe,
    summarize_subject_equal_active_edge_score,
)
from crychic.sender import SenderPrevalencePrior

from .certification import CrossFitOOFCertificationAudit
from .crossfit import CrossFitArtifacts

_SCHEMA_VERSION = "2.0.0"
_PRODUCER_MARKER = "crychic.workflow.crossfit_active_edge_point_records.v2"
_SOURCE_SEMANTICS = (
    "v4_cross_receiver_conserved_sender_lr_scores_complete_oof_only_v2"
)


def _contract_error(
    message: str,
    *,
    code: str,
    field: str,
    remediation: str,
) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=remediation,
    )


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _contrast_manifest_id(contrast: object) -> str:
    to_dict = getattr(contrast, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("registered cross-fit contrasts must provide to_dict()")
    identifier: str = stable_id("contrast_manifest", to_dict())
    return identifier


@dataclass(frozen=True, slots=True)
class _PointSource:
    contrast_name: str
    repeat_id: str
    score_version: str
    score_spec_id: str
    source_score_collection_id: str
    functional_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    sender_score_digests: tuple[str, ...]
    rows: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _FoldPointSource:
    fold_id: str
    application: CrossReceiverCommonScoringApplication
    candidate_drivers: tuple[tuple[tuple[str, str, str, str, str], str], ...]
    expected_rows: tuple[tuple[str, str, str], ...]
    table: pd.DataFrame


@dataclass(frozen=True, slots=True, init=False)
class CrossFitActiveEdgePointRecords:
    """Exact frozen-universe point records and their v4 OOF source lineage."""

    source_crossfit_id: str
    source_oof_audit_id: str
    source_registry_id: str
    active_edge_universe_id: str
    contrast_id: str
    contrast_name: str
    repeat_id: str
    score_version: str
    score_spec_id: str
    source_score_collection_id: str
    source_functional_ids: tuple[str, ...]
    source_application_ids: tuple[str, ...]
    source_sender_score_digests: tuple[str, ...]
    records: tuple[PointActiveEdgeScoreRecord, ...]
    collection_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "CrossFitActiveEdgePointRecords are producer-owned; use "
            "adapt_crossfit_active_edge_point_records()"
        )

    @property
    def point_records(self) -> tuple[PointActiveEdgeScoreRecord, ...]:
        """Return the candidate-complete inference records."""

        self._require_intact()
        return self.records

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_crossfit_id": self.source_crossfit_id,
            "source_oof_audit_id": self.source_oof_audit_id,
            "source_registry_id": self.source_registry_id,
            "active_edge_universe_id": self.active_edge_universe_id,
            "contrast_id": self.contrast_id,
            "contrast_name": self.contrast_name,
            "repeat_id": self.repeat_id,
            "score_version": self.score_version,
            "score_spec_id": self.score_spec_id,
            "source_score_collection_id": self.source_score_collection_id,
            "source_functional_ids": list(self.source_functional_ids),
            "source_application_ids": list(self.source_application_ids),
            "source_sender_score_digests": list(self.source_sender_score_digests),
            "point_record_ids": [record.record_id for record in self.records],
            "candidate_edge_ids": [record.candidate_edge_id for record in self.records],
            "source_semantics": _SOURCE_SEMANTICS,
            "producer_marker": _PRODUCER_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            for record in self.records:
                record._require_intact()
            candidate_ids = tuple(record.candidate_edge_id for record in self.records)
            expected_id = stable_id(
                "crossfit_active_edge_point_records",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and bool(self.records)
                and candidate_ids == tuple(sorted(candidate_ids))
                and len(candidate_ids) == len(set(candidate_ids))
                and {record.score_version for record in self.records}
                == {self.score_version}
                and {record.source_score_collection_id for record in self.records}
                == {self.source_score_collection_id}
                and self.collection_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Cross-fit active-edge point collection failed integrity validation",
                code="active_edge_point_collection_integrity_violation",
                field="collection_id",
                remediation="Re-adapt the intact complete OOF point artifacts",
            ) from error
        if not valid:
            raise _contract_error(
                "Cross-fit active-edge point collection failed integrity validation",
                code="active_edge_point_collection_integrity_violation",
                field="collection_id",
                remediation="Re-adapt the intact complete OOF point artifacts",
            )

    def to_dict(self) -> dict[str, object]:
        """Return an auditable representation without source score tables."""

        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "collection_id": self.collection_id,
            **payload,
            "records": [record.to_dict() for record in self.records],
        }


def _require_complete_oof_audit(
    artifacts: CrossFitArtifacts,
) -> CrossFitOOFCertificationAudit:
    audit = artifacts.oof_certification_audit
    if not isinstance(audit, CrossFitOOFCertificationAudit):
        raise TypeError("point OOF audit must be CrossFitOOFCertificationAudit")
    audit._require_intact()
    if (
        audit.source_crossfit_id != artifacts.crossfit_id
        or audit.source_registry_id != artifacts.receiver_scoring_registry_id
    ):
        raise _contract_error(
            "Point OOF audit does not belong to the source cross-fit artifacts",
            code="active_edge_point_oof_audit_mismatch",
            field="source_crossfit_id,source_registry_id",
            remediation="Recompute the OOF audit from the exact point artifacts",
        )
    if not audit.is_oof_descriptive_certified:
        raise _contract_error(
            "Active-edge point adaptation requires a complete descriptive OOF audit",
            code="active_edge_point_oof_not_certified",
            field="source_oof_audit_id",
            remediation="Complete every v4 train/apply requirement before adaptation",
        )
    return audit


def _registered_contrast(
    artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
) -> object:
    matched = tuple(
        contrast
        for contrast in artifacts.spec.contrasts
        if _contrast_manifest_id(contrast) == universe.contrast_id
    )
    if len(matched) != 1:
        raise _contract_error(
            "Frozen active-edge universe does not bind one registered contrast",
            code="active_edge_point_contrast_collection_mismatch",
            field="contrast_id",
            remediation="Freeze the universe with the exact contrast manifest ID",
        )
    return matched[0]


def _resolve_contrast(artifacts: CrossFitArtifacts, selector: str) -> object:
    requested = _name(selector, field_name="contrast_id_or_name")
    matched = tuple(
        contrast
        for contrast in artifacts.spec.contrasts
        if requested
        in {
            _contrast_manifest_id(contrast),
            _name(getattr(contrast, "name", None), field_name="contrast_name"),
        }
    )
    if len(matched) != 1:
        raise _contract_error(
            "Contrast selector is absent or ambiguous in the cross-fit specification",
            code="active_edge_point_contrast_collection_mismatch",
            field="contrast_id_or_name",
            remediation="Select one registered contrast name or manifest ID",
        )
    return matched[0]


def _source_key(row: object) -> tuple[str, str, str, str, str]:
    source = cast(Any, row)
    return (
        str(source.context_id),
        str(source.sender),
        str(source.receiver),
        str(source.interaction_id),
        str(source.mode),
    )


def _functional_driver_mapping(
    functional: CrossReceiverCommonScoringFunctional,
) -> dict[str, str]:
    functional._require_intact()
    driver_by_interaction: dict[str, str] = {}
    for interaction_id, _, driver_id in functional.interaction_mapping:
        interaction = _name(interaction_id, field_name="interaction_id")
        driver = _name(driver_id, field_name="driver_id")
        previous = driver_by_interaction.setdefault(interaction, driver)
        if previous != driver:
            raise _contract_error(
                "One fold maps an interaction to multiple target-prior drivers",
                code="active_edge_point_driver_mapping_conflict",
                field="interaction_id,driver_id",
                remediation="Refit the intact fold training candidate manifest",
            )
    return driver_by_interaction


def _functional_candidate_drivers(
    functional: CrossReceiverCommonScoringFunctional,
) -> dict[tuple[str, str, str, str, str], str]:
    driver_by_interaction = _functional_driver_mapping(functional)
    sender_functionals = tuple(
        child.sender_functional for child in functional.child_functionals
    )
    sender_ids = {item.sender_functional_id for item in sender_functionals}
    if len(sender_ids) != 1 or sender_ids != {functional.sender_functional_id}:
        raise _contract_error(
            "Cross-receiver children do not share one sender candidate manifest",
            code="active_edge_point_source_mismatch",
            field="sender_functional_id",
            remediation="Use one intact v4 cross-receiver functional",
        )
    sender_functional = sender_functionals[0]
    candidates: dict[tuple[str, str, str, str, str], str] = {}
    for prior in sender_functional.candidate_priors:
        if not isinstance(prior, SenderPrevalencePrior):
            raise TypeError(
                "candidate_priors must contain SenderPrevalencePrior values"
            )
        prior._require_intact()
        mapped_driver = driver_by_interaction.get(prior.interaction_id)
        if mapped_driver is None:
            raise _contract_error(
                "Sender candidate interaction is absent from the driver mapping",
                code="active_edge_point_source_mismatch",
                field="interaction_id",
                remediation="Use aligned sender and target-prior training manifests",
            )
        for context_id in functional.context_ids:
            for mode in (CommunicationMode.STATE, CommunicationMode.ECOSYSTEM):
                key = (
                    context_id,
                    prior.sender,
                    prior.receiver,
                    prior.interaction_id,
                    mode.value,
                )
                previous = candidates.setdefault(key, mapped_driver)
                if previous != mapped_driver:
                    raise _contract_error(
                        "One candidate edge maps to conflicting target-prior drivers",
                        code="active_edge_point_driver_mapping_conflict",
                        field="candidate_edge_id,driver_id",
                        remediation="Reject the inconsistent training manifests",
                    )
    if not candidates:
        raise _contract_error(
            "Training fold exposes no active-edge sender candidates",
            code="active_edge_point_universe_mismatch",
            field="candidate_priors",
            remediation="Fit a non-empty frozen sender candidate manifest",
        )
    return candidates


def _require_source_candidates(
    table: pd.DataFrame,
    candidate_drivers: dict[tuple[str, str, str, str, str], str],
    *,
    fold_id: str,
) -> None:
    observed: dict[tuple[str, str, str, str, str], set[str]] = {}
    for row in table.itertuples(index=False):
        observed.setdefault(_source_key(row), set()).add(str(row.driver_id))
    if not set(observed).issubset(candidate_drivers):
        raise _contract_error(
            "Fold sender scores contain an edge outside its training manifest",
            code="active_edge_point_universe_mismatch",
            field="candidate_edge_id",
            remediation="Use only the application-owned training candidate universe",
        )
    mismatched = tuple(
        key for key, drivers in observed.items() if drivers != {candidate_drivers[key]}
    )
    if mismatched:
        raise _contract_error(
            f"Fold {fold_id!r} sender rows use a different frozen driver mapping",
            code="active_edge_point_source_mismatch",
            field="driver_id",
            remediation="Use the exact v4 interaction-to-driver collection",
        )


def _design_contrast_id(contrast: object) -> str:
    to_dict = getattr(contrast, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("registered cross-fit contrasts must provide to_dict()")
    identifier: str = stable_id("contrast", to_dict())
    return identifier


def _expected_fold_rows(
    artifacts: CrossFitArtifacts,
    functional: CrossReceiverCommonScoringFunctional,
    contrast: object,
) -> tuple[tuple[str, str, str], ...]:
    sender_functional = functional.child_functionals[0].sender_functional
    context_by_token = {
        canonical_json(node): context_id
        for node, context_id in sender_functional.contrast_context_ids
    }
    coverage = artifacts.oof_coverage
    selected = coverage.loc[
        coverage["fold_id"].astype(str).eq(functional.fold_id)
        & coverage["contrast_id"].astype(str).eq(_design_contrast_id(contrast))
    ]
    rows: list[tuple[str, str, str]] = []
    for row in selected.itertuples(index=False):
        context_id = context_by_token.get(canonical_json(row.contrast_context))
        if context_id is None:
            raise _contract_error(
                "OOF coverage context is absent from the sender functional",
                code="active_edge_point_oof_audit_mismatch",
                field="contrast_context",
                remediation="Recompute coverage from the intact contrast functional",
            )
        rows.append((str(row.sample_id), str(row.subject_id), context_id))
    ordered = tuple(sorted(rows))
    if not ordered or len(ordered) != len(set(ordered)):
        raise _contract_error(
            "Fold OOF sample coverage is empty or duplicated",
            code="active_edge_point_oof_audit_mismatch",
            field="sample_id,subject_id,context_id",
            remediation="Use the exact complete OOF coverage table",
        )
    return ordered


def _missing_point_row(
    *,
    sample_id: str,
    subject_id: str,
    repeat_id: str,
    candidate: ActiveEdgeCandidate,
    score_version: str,
    reason_code: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "repeat_id": repeat_id,
        "context_id": candidate.context_id,
        "sender": candidate.sender,
        "receiver": candidate.receiver,
        "interaction_id": candidate.interaction_id,
        "mode": CommunicationMode(candidate.mode).value,
        "global_sender_lr_score": None,
        "status": ActiveEdgePointStatus.NOT_ESTIMABLE.value,
        "reason_code": reason_code,
        "score_version": score_version,
    }


def _complete_fold_candidate_rows(
    fold_source: _FoldPointSource,
    candidate: ActiveEdgeCandidate,
    *,
    repeat_id: str,
    score_version: str,
) -> pd.DataFrame:
    candidate_key = candidate.biological_key
    expected = tuple(
        (sample_id, subject_id)
        for sample_id, subject_id, context_id in fold_source.expected_rows
        if context_id == candidate.context_id
    )
    if not expected:
        raise _contract_error(
            "Candidate context has no expected held-out sample in one fold",
            code="active_edge_point_oof_audit_mismatch",
            field="context_id",
            remediation="Use an estimable fold with complete context coverage",
        )
    candidate_drivers = dict(fold_source.candidate_drivers)
    selected = _candidate_rows(
        fold_source.table,
        context_id=candidate.context_id,
        sender=candidate.sender,
        receiver=candidate.receiver,
        interaction_id=candidate.interaction_id,
        mode=CommunicationMode(candidate.mode),
    )
    observed_pairs = tuple(
        selected.loc[:, ["sample_id", "subject_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if len(observed_pairs) != len(set(observed_pairs)) or not set(
        observed_pairs
    ).issubset(expected):
        raise _contract_error(
            "Candidate score rows duplicate or exceed the expected OOF sample grid",
            code="active_edge_point_oof_audit_mismatch",
            field="sample_id,subject_id",
            remediation="Use the exact application-owned held-out rows",
        )
    selected = selected.loc[
        :,
        [
            "sample_id",
            "subject_id",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "global_sender_lr_score",
            "status",
            "reason_code",
            "score_version",
        ],
    ].copy(deep=True)
    selected.insert(2, "repeat_id", repeat_id)
    missing_reason = (
        "active_edge_candidate_not_in_training_fold"
        if candidate_key not in candidate_drivers
        else "active_edge_candidate_missing_from_fold_scores"
    )
    missing = [
        _missing_point_row(
            sample_id=sample_id,
            subject_id=subject_id,
            repeat_id=repeat_id,
            candidate=candidate,
            score_version=score_version,
            reason_code=missing_reason,
        )
        for sample_id, subject_id in expected
        if (sample_id, subject_id) not in set(observed_pairs)
    ]
    if missing:
        selected = pd.concat((selected, pd.DataFrame(missing)), ignore_index=True)
    return selected


def _point_source(
    artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
    audit: CrossFitOOFCertificationAudit,
    contrast: object,
) -> _PointSource:
    contrast_name = _name(getattr(contrast, "name", None), field_name="contrast_name")
    repeat_id = _name(artifacts.spec.repeat_id, field_name="repeat_id")
    universe_drivers = {
        candidate.biological_key: candidate.driver_id
        for candidate in universe.candidates
    }
    selected: list[_FoldPointSource] = []
    for fold in artifacts.folds:
        fold_id = _name(fold.fold_id, field_name="fold_id")
        registered_functionals = tuple(
            functional
            for functional in fold.cross_receiver_common_functionals
            if isinstance(functional, CrossReceiverCommonScoringFunctional)
            and functional.contrast_manifest_id == universe.contrast_id
        )
        if len(registered_functionals) != 1:
            raise _contract_error(
                "Each fold must register one unique v4 contrast functional",
                code="active_edge_point_contrast_collection_mismatch",
                field="cross_receiver_common_functionals",
                remediation=(
                    "Retain the exact training functional used by the application"
                ),
            )
        registered_functional = registered_functionals[0]
        registered_functional._require_intact()
        applications = tuple(
            application
            for application in fold.cross_receiver_common_applications
            if isinstance(application, CrossReceiverCommonScoringApplication)
            and application.functional.contrast_manifest_id == universe.contrast_id
        )
        if len(applications) != 1:
            raise _contract_error(
                "Each fold must expose one unique v4 contrast score application",
                code="active_edge_point_contrast_collection_mismatch",
                field="cross_receiver_common_applications",
                remediation=(
                    "Run one cross-receiver v4 application per fold and contrast"
                ),
            )
        application = applications[0]
        application._require_intact()
        functional = application.functional
        if not isinstance(functional, CrossReceiverCommonScoringFunctional):
            raise TypeError("v4 point application must own a typed functional")
        functional._require_intact()
        if (
            functional.fold_id != fold_id
            or functional.contrast_name != contrast_name
            or functional.contrast_manifest_id != universe.contrast_id
            or functional.global_common_functional_id
            != registered_functional.global_common_functional_id
        ):
            raise _contract_error(
                "Cross-receiver application has the wrong registered lineage",
                code="active_edge_point_source_mismatch",
                field="fold_id,contrast_id,global_common_functional_id",
                remediation="Use the exact application selected from this fold",
            )
        table = application.global_sender_lr_scores
        if tuple(table.columns) != GLOBAL_COMMON_SENDER_SCORE_COLUMNS:
            raise _contract_error(
                "Active-edge point source is not the v4 global sender-LR table",
                code="active_edge_point_source_table_mismatch",
                field="global_sender_lr_scores",
                remediation=(
                    "Use global_sender_lr_scores; legacy comm_strength and "
                    "receiver-local scores are forbidden"
                ),
            )
        if table.empty:
            raise _contract_error(
                "Active-edge point source table is empty",
                code="active_edge_point_universe_mismatch",
                field="global_sender_lr_scores",
                remediation="Emit every frozen candidate with a typed row",
            )
        if set(table["global_common_functional_id"].astype(str)) != {
            functional.global_common_functional_id
        }:
            raise _contract_error(
                "Sender score table does not belong to its v4 functional",
                code="active_edge_point_source_mismatch",
                field="global_common_functional_id",
                remediation="Use the intact application-owned sender score table",
            )
        if set(table["score_version"].astype(str)) != {functional.score_version}:
            raise _contract_error(
                "Sender score table and functional score versions differ",
                code="active_edge_point_source_mismatch",
                field="score_version",
                remediation="Reapply the intact v4 scoring functional",
            )
        candidate_drivers = _functional_candidate_drivers(functional)
        if not set(candidate_drivers).issubset(universe_drivers):
            raise _contract_error(
                "Fold training candidates exceed the authoritative universe",
                code="active_edge_point_universe_mismatch",
                field="candidate_edge_id",
                remediation=(
                    "Adapt only child manifests that are subsets of the point universe"
                ),
            )
        if any(
            universe_drivers[key] != driver_id
            for key, driver_id in candidate_drivers.items()
        ):
            raise _contract_error(
                "Fold and authoritative universe driver mappings disagree",
                code="active_edge_point_driver_mapping_conflict",
                field="candidate_edge_id,driver_id",
                remediation="Use the exact point-universe target-prior mapping",
            )
        _require_source_candidates(table, candidate_drivers, fold_id=fold_id)
        selected.append(
            _FoldPointSource(
                fold_id=fold_id,
                application=application,
                candidate_drivers=tuple(sorted(candidate_drivers.items())),
                expected_rows=_expected_fold_rows(
                    artifacts,
                    functional,
                    contrast,
                ),
                table=table.copy(deep=True),
            )
        )

    if not selected or len(selected) != len(artifacts.folds):
        raise _contract_error(
            "Active-edge point source does not cover every outer fold",
            code="active_edge_point_contrast_collection_mismatch",
            field="folds",
            remediation="Retain one v4 contrast application for every fold",
        )
    selected.sort(key=lambda item: item.fold_id)
    functionals = tuple(item.application.functional for item in selected)
    score_versions = {functional.score_version for functional in functionals}
    score_specs = {functional.spec.spec_id for functional in functionals}
    if score_versions != {universe.score_version} or len(score_specs) != 1:
        raise _contract_error(
            "Point applications do not share the frozen universe score contract",
            code="active_edge_point_score_spec_mismatch",
            field="score_version,score_spec_id",
            remediation="Freeze one universe for one v4 score version and spec",
        )
    score_spec_id = next(iter(score_specs))
    source_entries = [
        {
            "fold_id": item.fold_id,
            "global_common_functional_id": (
                item.application.functional.global_common_functional_id
            ),
            "application_id": item.application.application_id,
            "sender_scores_digest": item.application.sender_scores_digest,
        }
        for item in selected
    ]
    source_collection_id = stable_id(
        "active_edge_point_score_collection",
        {
            "source_crossfit_id": artifacts.crossfit_id,
            "source_oof_audit_id": audit.audit_id,
            "source_registry_id": artifacts.receiver_scoring_registry_id,
            "contrast_id": universe.contrast_id,
            "contrast_name": contrast_name,
            "repeat_id": repeat_id,
            "score_version": universe.score_version,
            "score_spec_id": score_spec_id,
            "fold_sources": source_entries,
            "source_semantics": _SOURCE_SEMANTICS,
        },
        schema_version=_SCHEMA_VERSION,
    )
    completed = tuple(
        _complete_fold_candidate_rows(
            fold_source,
            candidate,
            repeat_id=repeat_id,
            score_version=universe.score_version,
        )
        for fold_source in selected
        for candidate in universe.candidates
    )
    rows = pd.concat(completed, ignore_index=True)
    return _PointSource(
        contrast_name=contrast_name,
        repeat_id=repeat_id,
        score_version=universe.score_version,
        score_spec_id=score_spec_id,
        source_score_collection_id=source_collection_id,
        functional_ids=tuple(
            item.application.functional.global_common_functional_id for item in selected
        ),
        application_ids=tuple(item.application.application_id for item in selected),
        sender_score_digests=tuple(
            item.application.sender_scores_digest for item in selected
        ),
        rows=rows,
    )


def _candidate_rows(
    source: pd.DataFrame,
    *,
    context_id: str,
    sender: str,
    receiver: str,
    interaction_id: str,
    mode: CommunicationMode,
) -> pd.DataFrame:
    mask = (
        source["context_id"].astype(str).eq(context_id)
        & source["sender"].astype(str).eq(sender)
        & source["receiver"].astype(str).eq(receiver)
        & source["interaction_id"].astype(str).eq(interaction_id)
        & source["mode"].astype(str).eq(mode.value)
    )
    return source.loc[mask].copy(deep=True)


def _point_record(
    point: SubjectEqualActiveEdgeScore,
) -> PointActiveEdgeScoreRecord:
    point._require_intact()
    return PointActiveEdgeScoreRecord(
        candidate_edge_id=point.candidate_edge_id,
        stratum_id=point.stratum_id,
        score_version=point.score_version,
        source_score_collection_id=point.source_collection_id,
        n_subjects=point.n_subjects,
        score=point.score,
        status=ActiveEdgeScoreStatus(ActiveEdgePointStatus(point.status).value),
        reason_code=point.reason_code,
    )


def freeze_crossfit_active_edge_universe(
    point_artifacts: CrossFitArtifacts,
    *,
    contrast_id_or_name: str,
    universe_name: str | None = None,
) -> FrozenActiveEdgeUniverse:
    """Freeze the union of training-fold v4 sender manifests without score access."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    point_artifacts._require_intact()
    contrast = _resolve_contrast(point_artifacts, contrast_id_or_name)
    contrast_id = _contrast_manifest_id(contrast)
    contrast_name = _name(getattr(contrast, "name", None), field_name="contrast_name")
    fold_candidates: list[dict[tuple[str, str, str, str, str], str]] = []
    global_driver_by_interaction: dict[str, str] = {}
    score_versions: set[str] = set()
    score_spec_ids: set[str] = set()
    for fold in point_artifacts.folds:
        functionals = tuple(
            functional
            for functional in fold.cross_receiver_common_functionals
            if isinstance(functional, CrossReceiverCommonScoringFunctional)
            and functional.contrast_manifest_id == contrast_id
        )
        if len(functionals) != 1:
            raise _contract_error(
                "Each fold must expose one unique v4 contrast training functional",
                code="active_edge_point_contrast_collection_mismatch",
                field="cross_receiver_common_functionals",
                remediation=(
                    "Fit one v4 cross-receiver functional per fold and contrast"
                ),
            )
        functional = functionals[0]
        functional._require_intact()
        if functional.contrast_name != contrast_name:
            raise _contract_error(
                "Cross-receiver functional has the wrong registered contrast name",
                code="active_edge_point_source_mismatch",
                field="contrast_name",
                remediation="Use the exact registered contrast functional",
            )
        for interaction_id, driver_id in _functional_driver_mapping(functional).items():
            previous = global_driver_by_interaction.setdefault(
                interaction_id, driver_id
            )
            if previous != driver_id:
                raise _contract_error(
                    "Fold training manifests disagree on an interaction driver",
                    code="active_edge_point_driver_mapping_conflict",
                    field="interaction_id,driver_id",
                    remediation=("Reject the inconsistent fold target-prior mappings"),
                )
        fold_candidates.append(_functional_candidate_drivers(functional))
        score_versions.add(functional.score_version)
        score_spec_ids.add(functional.spec.spec_id)
    if len(score_versions) != 1 or len(score_spec_ids) != 1:
        raise _contract_error(
            "Training folds do not share one v4 active-edge score contract",
            code="active_edge_point_score_spec_mismatch",
            field="score_version,score_spec_id",
            remediation="Refit all folds with one frozen cross-receiver score spec",
        )
    union: dict[tuple[str, str, str, str, str], str] = {}
    for candidates in fold_candidates:
        for key, driver_id in candidates.items():
            previous = union.setdefault(key, driver_id)
            if previous != driver_id:
                raise _contract_error(
                    "Fold training manifests disagree on an edge driver mapping",
                    code="active_edge_point_driver_mapping_conflict",
                    field="candidate_edge_id,driver_id",
                    remediation="Reject the inconsistent fold target-prior mappings",
                )
    if not union:
        raise _contract_error(
            "Cross-fit training folds expose no active-edge candidates",
            code="active_edge_point_universe_mismatch",
            field="candidate_priors",
            remediation="Fit non-empty v4 sender candidate manifests",
        )
    frozen_candidates = tuple(
        ActiveEdgeCandidate(
            contrast_id=contrast_id,
            context_id=key[0],
            sender=key[1],
            receiver=key[2],
            interaction_id=key[3],
            driver_id=driver_id,
            mode=key[4],
        )
        for key, driver_id in sorted(union.items())
    )
    name = (
        f"{contrast_name}-active-edge-universe-v1"
        if universe_name is None
        else _name(universe_name, field_name="universe_name")
    )
    return freeze_active_edge_universe(
        frozen_candidates,
        universe_name=name,
        contrast_id=contrast_id,
        score_version=next(iter(score_versions)),
    )


def _adapt_verified_crossfit_active_edge_records(
    point_artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
    audit: CrossFitOOFCertificationAudit,
    contrast: object,
) -> CrossFitActiveEdgePointRecords:
    source = _point_source(point_artifacts, universe, audit, contrast)

    records: list[PointActiveEdgeScoreRecord] = []
    for candidate in universe.candidates:
        rows = _candidate_rows(
            source.rows,
            context_id=candidate.context_id,
            sender=candidate.sender,
            receiver=candidate.receiver,
            interaction_id=candidate.interaction_id,
            mode=CommunicationMode(candidate.mode),
        )
        point = summarize_subject_equal_active_edge_score(
            candidate,
            rows,
            source_collection_id=source.source_score_collection_id,
            score_version=source.score_version,
        )
        if point.stratum_id != candidate.stratum_id(
            score_version=universe.score_version
        ):
            raise _contract_error(
                "Point reducer returned the wrong frozen probability stratum",
                code="active_edge_point_source_mismatch",
                field="stratum_id",
                remediation="Use one universe and score version for point adaptation",
            )
        records.append(_point_record(point))
    ordered = tuple(sorted(records, key=lambda record: record.candidate_edge_id))
    if tuple(record.candidate_edge_id for record in ordered) != (
        universe.candidate_edge_ids
    ):
        raise _contract_error(
            "Point adapter did not emit the exact frozen candidate universe",
            code="active_edge_point_universe_mismatch",
            field="candidate_edge_id",
            remediation="Emit one point record for every frozen candidate",
        )

    self = object.__new__(CrossFitActiveEdgePointRecords)
    values: dict[str, object] = {
        "source_crossfit_id": point_artifacts.crossfit_id,
        "source_oof_audit_id": audit.audit_id,
        "source_registry_id": point_artifacts.receiver_scoring_registry_id,
        "active_edge_universe_id": universe.universe_id,
        "contrast_id": universe.contrast_id,
        "contrast_name": source.contrast_name,
        "repeat_id": source.repeat_id,
        "score_version": source.score_version,
        "score_spec_id": source.score_spec_id,
        "source_score_collection_id": source.source_score_collection_id,
        "source_functional_ids": source.functional_ids,
        "source_application_ids": source.application_ids,
        "source_sender_score_digests": source.sender_score_digests,
        "records": ordered,
        "_producer_marker": _PRODUCER_MARKER,
    }
    for field_name, value in values.items():
        object.__setattr__(self, field_name, value)
    object.__setattr__(
        self,
        "collection_id",
        stable_id(
            "crossfit_active_edge_point_records",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    self._require_intact()
    return self


def adapt_crossfit_active_edge_records_against_universe(
    point_artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
) -> CrossFitActiveEdgePointRecords:
    """Adapt v4 OOF rows against an existing authoritative candidate universe.

    Fold training manifests may be strict subsets of ``universe``. Those absent
    fold candidates are emitted as explicit not-estimable source rows before the
    repeat- and subject-equal reducer runs.
    """

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(universe, FrozenActiveEdgeUniverse):
        raise TypeError("universe must be FrozenActiveEdgeUniverse")
    point_artifacts._require_intact()
    universe._require_intact()
    audit = _require_complete_oof_audit(point_artifacts)
    contrast = _registered_contrast(point_artifacts, universe)
    return _adapt_verified_crossfit_active_edge_records(
        point_artifacts,
        universe,
        audit,
        contrast,
    )


def adapt_crossfit_active_edge_point_records(
    point_artifacts: CrossFitArtifacts,
    universe: FrozenActiveEdgeUniverse,
) -> CrossFitActiveEdgePointRecords:
    """Strictly adapt a point workflow's exact source-derived universe."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(universe, FrozenActiveEdgeUniverse):
        raise TypeError("universe must be FrozenActiveEdgeUniverse")
    point_artifacts._require_intact()
    universe._require_intact()
    audit = _require_complete_oof_audit(point_artifacts)
    contrast = _registered_contrast(point_artifacts, universe)
    expected_universe = freeze_crossfit_active_edge_universe(
        point_artifacts,
        contrast_id_or_name=universe.contrast_id,
        universe_name=universe.universe_name,
    )
    if expected_universe.universe_id != universe.universe_id:
        raise _contract_error(
            "Supplied universe is not the complete training-manifest union",
            code="active_edge_point_universe_mismatch",
            field="active_edge_universe_id",
            remediation="Freeze the universe directly from the intact point artifacts",
        )
    return _adapt_verified_crossfit_active_edge_records(
        point_artifacts,
        universe,
        audit,
        contrast,
    )


__all__ = [
    "CrossFitActiveEdgePointRecords",
    "adapt_crossfit_active_edge_point_records",
    "adapt_crossfit_active_edge_records_against_universe",
    "freeze_crossfit_active_edge_universe",
]
