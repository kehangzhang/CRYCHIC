"""Family-first held-out attribution and contrast-common mechanistic scoring.

This module composes already-frozen receiver-family, incremental-downstream,
and common-sender parents.  It deliberately does not fit another response
model.  Held-out response evidence selects families; response-independent
member evidence allocates an estimable family score; sender weights only split
an already-established member score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from crychic.attribution import ReceiverFamilyTrainingArtifact
from crychic.core import ContractError, stable_id
from crychic.core._validation import (
    record_validation,
    validation_is_cached,
    validation_scope,
)
from crychic.sender import (
    CommonSenderApplication,
    ContrastCommonSenderFunctional,
    SenderContrastSupportStatus,
    interaction_ligand_contrast_gates,
)

from .contracts import float64_array_digest
from .downstream import (
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
)
from .integration import pair_softmin
from .receiver_program import (
    ReceiverProgramApplication,
    ReceiverProgramTrainingArtifact,
)

FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS = (
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "interaction_id",
    "mode",
    "ligand_contrast_gate",
    "ligand_contrast_gate_id",
    "ligand_contrast_gate_status",
    "ligand_contrast_gate_reason_code",
    "availability",
    "ligand_availability",
    "prior_quality",
    "subject_prevalence",
    "resource_evidence",
)

FAMILY_HELDOUT_ATTRIBUTION_COLUMNS = (
    "family_common_functional_id",
    "family_id",
    "family_coefficient",
    "family_estimable",
    "family_gain",
    "raw_family_gain",
    "family_selected",
    "family_selection_frequency",
    "selection_status",
    "reason_code",
)

FAMILY_SUBJECT_DIFFERENTIAL_COLUMNS = (
    "family_common_functional_id",
    "subject_id",
    "family_id",
    "null_loss",
    "family_loss",
    "differential_effect",
    "bounded_incremental_gain",
    "status",
    "reason_code",
)

FAMILY_COMMON_SCORE_COLUMNS = (
    "family_common_functional_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "mode",
    "availability_score",
    "family_availability",
    "receptor_eligible",
    "ligand_contrast_gate_status",
    "ligand_contrast_supported_interaction_count",
    "ligand_contrast_not_estimable_interaction_count",
    "receiver_program_score",
    "receiver_program_status",
    "receiver_program_reason_code",
    "incremental_downstream_gain",
    "differential_effect",
    "family_coefficient",
    "family_selected",
    "within_family_entropy",
    "lr_identifiability_status",
    "family_core_strength",
    "integrated_lr_score",
    "status",
    "reason_code",
    "score_version",
)

FAMILY_MEMBER_SCORE_COLUMNS = (
    "family_common_functional_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "availability",
    "receptor_gate",
    "receptor_eligible",
    "ligand_contrast_gate",
    "ligand_contrast_gate_id",
    "ligand_contrast_gate_status",
    "ligand_contrast_gate_reason_code",
    "ligand_availability",
    "prior_quality",
    "subject_prevalence",
    "resource_evidence",
    "member_evidence_score",
    "within_family_lr_weight",
    "within_family_entropy",
    "lr_identifiability_status",
    "family_core_strength",
    "sender_unresolved_strength",
    "status",
    "reason_code",
    "score_version",
)

FAMILY_SENDER_SCORE_COLUMNS = (
    "family_common_functional_id",
    "sender_functional_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
    "sender_unresolved_strength",
    "assignment_weight",
    "sender_resolved_strength",
    "status",
    "reason_code",
    "score_version",
)

_FUNCTIONAL_PRODUCER = "crychic.family_common_scoring_functional.v2"
_APPLICATION_PRODUCER = "crychic.family_common_scoring_application.v2"
FAMILY_COMMON_SCORE_VERSION: str = (
    "family_first_mechanistic_ligand_contrast_gated_softmin_v2"
)
_SCORE_VERSION = FAMILY_COMMON_SCORE_VERSION
_CERTIFICATION_STATUS = "heldout_family_common_diagnostic_not_oof_certified_v1"
_SELECTION_FREQUENCY_REASON = "single_heldout_application_no_resampling"
_RELEASED_MODES = ("ecosystem", "state")
_MEMBER_ALLOCATION_METHOD = (
    "availability_x_hard_receptor_eligibility_x_frozen_ligand_contrast_gate_x_"
    "frozen_receptor_gate_x_"
    "ligand_x_prior_quality_x_subject_prevalence_x_resource_evidence_"
    "then_supported_family_normalize_v2"
)


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _unit_value(value: object, *, field_name: str) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be numeric or missing") from error
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must lie in [0, 1] or be missing")
    return numeric


def _canonical_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _canonical_scalar(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _table_digest(name: str, table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    if tuple(table.columns) != columns:
        raise ValueError(f"{name} columns do not match the released contract")
    rows = [
        [_canonical_scalar(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    return str(
        stable_id(
            name,
            {"columns": list(columns), "rows": rows},
            schema_version="1",
        )
    )


def family_common_edge_evidence_digest(edge_evidence: pd.DataFrame) -> str:
    """Digest canonical edge evidence consumed by a scoring application."""

    if not isinstance(edge_evidence, pd.DataFrame):
        raise TypeError("edge_evidence must be a pandas DataFrame")
    if set(edge_evidence.columns) != set(FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS):
        raise ValueError("edge_evidence columns do not match the released contract")
    identifiers = (
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
        "ligand_contrast_gate_id",
    )
    numeric_columns = (
        "ligand_contrast_gate",
        "availability",
        "ligand_availability",
        "prior_quality",
        "subject_prevalence",
        "resource_evidence",
    )
    key = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    canonical = edge_evidence.loc[
        :, list(FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS)
    ].copy()
    for column in identifiers:
        canonical[column] = [
            _required_name(value, field_name=column) for value in canonical[column]
        ]
    for column in numeric_columns:
        canonical[column] = [
            _unit_value(value, field_name=column) for value in canonical[column]
        ]
    canonical["ligand_contrast_gate_status"] = [
        SenderContrastSupportStatus(value).value
        for value in canonical["ligand_contrast_gate_status"]
    ]
    canonical["ligand_contrast_gate_reason_code"] = pd.Series(
        [
            None
            if value is None or value is pd.NA or pd.isna(value)
            else _required_name(
                value, field_name="ligand_contrast_gate_reason_code"
            )
            for value in canonical["ligand_contrast_gate_reason_code"]
        ],
        index=canonical.index,
        dtype=object,
    )
    canonical = canonical.sort_values(key, kind="stable", ignore_index=True)
    return _table_digest(
        "family_common_edge_evidence_input",
        canonical,
        FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS,
    )


def family_common_sender_application_digest(
    sender_application: CommonSenderApplication,
) -> str:
    """Digest the exact common-sender rows consumed by one application."""

    if not isinstance(sender_application, CommonSenderApplication):
        raise TypeError("sender_application must be CommonSenderApplication")
    sender_application._require_intact()
    return _table_digest(
        "family_common_sender_application_input",
        sender_application.table,
        tuple(sender_application.table.columns),
    )


def _sparse_matrix_id(matrix: sparse.spmatrix) -> str:
    canonical = sparse.csc_matrix(matrix, dtype=np.float64, copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    return str(
        stable_id(
            "sparse_matrix",
            {
                "data_digest": float64_array_digest(canonical.data),
                "indices": canonical.indices.astype(int).tolist(),
                "indptr": canonical.indptr.astype(int).tolist(),
                "shape": list(canonical.shape),
            },
        )
    )


@dataclass(frozen=True, slots=True)
class FrozenFamilyInteraction:
    """One interaction mapped to a frozen driver family and receptor gate."""

    interaction_id: str
    driver_id: str
    family_id: str
    receptor_gate: float
    receptor_eligible: bool

    def __post_init__(self) -> None:
        for field_name in ("interaction_id", "driver_id", "family_id"):
            _required_name(getattr(self, field_name), field_name=field_name)
        gate = float(self.receptor_gate)
        if not math.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError("receptor_gate must lie in [0, 1]")
        if not isinstance(self.receptor_eligible, bool):
            raise TypeError("receptor_eligible must be boolean")
        object.__setattr__(self, "receptor_gate", gate)

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "driver_id": self.driver_id,
            "family_id": self.family_id,
            "receptor_gate": self.receptor_gate,
            "receptor_eligible": self.receptor_eligible,
        }


@dataclass(frozen=True, slots=True, init=False)
class FamilyCommonScoringFunctional:
    """One producer-owned scoring function for a ContrastSpec by fold."""

    receiver_family: ReceiverFamilyTrainingArtifact
    receiver_program_artifact: ReceiverProgramTrainingArtifact | None
    incremental_functional: IncrementalDownstreamFunctional | None
    sender_functional: ContrastCommonSenderFunctional
    receiver: str
    contrast_name: str
    contrast_manifest_id: str
    fold_id: str
    context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    filter_universe_id: str
    family_ids: tuple[str, ...]
    active_family_ids: tuple[str, ...]
    interactions: tuple[FrozenFamilyInteraction, ...]
    receiver_incremental_training_artifact_id: str
    tuning_manifest_id: str
    selected_penalty_id: str | None
    autonomous_program_resource_id: str | None
    receiver_program_reason_code: str | None
    incremental_reason_code: str | None
    family_selection_threshold: float
    softmin_power: float
    epsilon: float
    score_version: str
    certification_status: str
    family_common_functional_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FamilyCommonScoringFunctional is producer-owned; use "
            "fit_family_common_scoring_functional()"
        )

    @property
    def interaction_ids(self) -> tuple[str, ...]:
        return tuple(item.interaction_id for item in self.interactions)

    @property
    def autonomous_program_source_id(self) -> str | None:
        """Return the effective static, learned, or composed nuisance basis ID."""

        return self.autonomous_program_resource_id

    @property
    def is_oof_certified(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        interaction_gates = interaction_ligand_contrast_gates(
            self.sender_functional,
            tuple(
                (self.receiver, interaction.interaction_id)
                for interaction in self.interactions
            ),
        )
        return {
            "active_family_ids": list(self.active_family_ids),
            "certification_status": self.certification_status,
            "contrast_manifest_id": self.contrast_manifest_id,
            "contrast_name": self.contrast_name,
            "context_ids": list(self.context_ids),
            "epsilon": self.epsilon,
            "family_ids": list(self.family_ids),
            "family_selection_threshold": self.family_selection_threshold,
            "filter_universe_id": self.filter_universe_id,
            "fold_id": self.fold_id,
            "incremental_functional_id": (
                None
                if self.incremental_functional is None
                else self.incremental_functional.incremental_functional_id
            ),
            "incremental_reason_code": self.incremental_reason_code,
            "interactions": [item.to_dict() for item in self.interactions],
            "interaction_ligand_contrast_gates": [
                gate.to_dict() for gate in interaction_gates
            ],
            "member_allocation_method": _MEMBER_ALLOCATION_METHOD,
            "receiver": self.receiver,
            "receiver_program_training_artifact_id": (
                None
                if self.receiver_program_artifact is None
                else self.receiver_program_artifact.training_artifact_id
            ),
            "receiver_program_reason_code": self.receiver_program_reason_code,
            "receiver_incremental_training_artifact_id": (
                self.receiver_incremental_training_artifact_id
            ),
            "receiver_family_training_artifact_id": (
                self.receiver_family.training_artifact_id
            ),
            "score_version": self.score_version,
            "sender_functional_id": self.sender_functional.sender_functional_id,
            "selected_penalty_id": self.selected_penalty_id,
            "softmin_power": self.softmin_power,
            "training_subject_ids": list(self.training_subject_ids),
            "tuning_manifest_id": self.tuning_manifest_id,
            "autonomous_program_resource_id": self.autonomous_program_resource_id,
        }

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            self.receiver_family._require_producer_owned()
            self.sender_functional._require_intact()
            if self.receiver_program_artifact is None:
                if not self.receiver_program_reason_code:
                    raise ValueError(
                        "missing receiver-program parent requires a reason"
                    )
            else:
                self.receiver_program_artifact._require_intact()
                _validate_receiver_program_parent(
                    self.receiver_family,
                    self.receiver_program_artifact,
                    self.sender_functional,
                )
                if self.receiver_program_reason_code != (
                    self.receiver_program_artifact.reason_code
                ):
                    raise ValueError("receiver-program parent reason changed")
            if self.incremental_functional is None:
                if not self.incremental_reason_code:
                    raise ValueError("missing incremental functional requires a reason")
                if (
                    self.active_family_ids != self.receiver_family.eligible_family_ids
                    or self.training_subject_ids
                    != self.receiver_family.training_subject_ids
                ):
                    raise ValueError("not-estimable functional lineage changed")
            else:
                if self.incremental_reason_code is not None:
                    raise ValueError(
                        "observed incremental functional cannot have a reason"
                    )
                self.incremental_functional._require_intact()
                _validate_parent_lineage(
                    self.receiver_family,
                    self.incremental_functional,
                    self.sender_functional,
                    autonomous_program_resource_id=(
                        self.autonomous_program_resource_id
                    ),
                )
            if (
                self._producer_marker != _FUNCTIONAL_PRODUCER
                or self.score_version != _SCORE_VERSION
                or self.certification_status != _CERTIFICATION_STATUS
                or self.is_oof_certified
                or tuple(
                    sorted(self.interactions, key=lambda item: item.interaction_id)
                )
                != self.interactions
                or self.family_ids != self.receiver_family.family_basis.family_ids
                or (
                    self.incremental_functional is not None
                    and self.active_family_ids != self.incremental_functional.family_ids
                )
                or self.context_ids != self.sender_functional.context_ids
                or any(
                    not _required_name(getattr(self, field_name), field_name=field_name)
                    for field_name in (
                        "receiver_incremental_training_artifact_id",
                        "tuning_manifest_id",
                    )
                )
                or (
                    self.incremental_functional is not None
                    and not _required_name(
                        self.selected_penalty_id,
                        field_name="selected_penalty_id",
                    )
                )
                or (
                    self.autonomous_program_resource_id is not None
                    and not _required_name(
                        self.autonomous_program_resource_id,
                        field_name="autonomous_program_resource_id",
                    )
                )
                or stable_id(
                    "family_common_scoring_functional",
                    self._identity_payload(),
                    schema_version="2",
                )
                != self.family_common_functional_id
            ):
                raise ValueError("family-common functional identity changed")
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Family-common scoring functional failed integrity validation",
                code="family_common_functional_integrity_violation",
                field="family_common_functional_id",
                remediation=(
                    "Refit from intact receiver, incremental, and sender parents"
                ),
            ) from error
        record_validation(self)

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "family_common_functional_id": self.family_common_functional_id,
            **self._identity_payload(),
            "autonomous_program_source_id": self.autonomous_program_source_id,
            "common_across_contexts": True,
            "family_first": True,
            "sender_is_allocation_only": True,
            "member_allocation_reuses_mechanistic_evidence": True,
            "member_allocation_reuse_scope": (
                "within_family_weights_only_not_family_core_multiplication"
            ),
            "edge_evidence_provenance_status": "caller_supplied_unverified",
            "differential_effect_estimand": (
                "subject_receiver_null_vs_single_family_loss_ratio_v1"
            ),
            "receiver_program_status": (
                "not_estimable"
                if self.receiver_program_artifact is None
                else self.receiver_program_artifact.status
            ),
            "receiver_program_reason_code": self.receiver_program_reason_code,
            "is_oof_certified": self.is_oof_certified,
        }


def _expected_original_family_basis_id(
    receiver_family: ReceiverFamilyTrainingArtifact,
) -> str:
    basis = receiver_family.family_basis
    indices = np.flatnonzero(basis.family_eligible)
    return _sparse_matrix_id(basis.matrix[:, indices])


def _validate_parent_lineage(
    receiver_family: ReceiverFamilyTrainingArtifact,
    incremental_functional: IncrementalDownstreamFunctional,
    sender_functional: ContrastCommonSenderFunctional,
    *,
    autonomous_program_resource_id: str | None,
) -> None:
    expected_active = receiver_family.eligible_family_ids
    expected = (
        receiver_family.receiver,
        receiver_family.fold_id,
        receiver_family.training_subject_ids,
        receiver_family.filter_universe_id,
        expected_active,
        _expected_original_family_basis_id(receiver_family),
        autonomous_program_resource_id,
    )
    observed = (
        incremental_functional.receiver,
        incremental_functional.fold_id,
        incremental_functional.training_subject_ids,
        sender_functional.filter_universe_id,
        incremental_functional.family_ids,
        incremental_functional.original_family_basis_id,
        incremental_functional.autonomous_basis_id,
    )
    if observed != expected:
        raise ContractError(
            "Family-common parents do not share one receiver/fold/family universe",
            code="family_common_parent_mismatch",
            field="parent_artifacts",
            remediation="Use parents produced inside the same physical training fold",
        )
    if (
        sender_functional.contrast_name != incremental_functional.contrast_name
        or sender_functional.training_subject_ids
        != incremental_functional.training_subject_ids
    ):
        raise ContractError(
            "Family-common parents do not share one contrast and training scope",
            code="family_common_parent_mismatch",
            field="contrast_manifest_id",
            remediation="Use the contrast-common sender functional from the same fold",
        )


def _validate_receiver_program_parent(
    receiver_family: ReceiverFamilyTrainingArtifact,
    receiver_program: ReceiverProgramTrainingArtifact,
    sender_functional: ContrastCommonSenderFunctional,
) -> None:
    """Bind target-program semantics without consulting receptor eligibility."""

    receiver_program._require_intact()
    expected = (
        receiver_family.training_artifact_id,
        receiver_family.receiver,
        receiver_family.fold_id,
        receiver_family.training_subject_ids,
        receiver_family.family_basis.family_ids,
        receiver_family.source_basis.feature_ids,
        sender_functional.contrast_name,
    )
    observed = (
        receiver_program.receiver_family_artifact.training_artifact_id,
        receiver_program.receiver,
        receiver_program.fold_id,
        receiver_program.training_subject_ids,
        receiver_program.family_ids,
        receiver_program.feature_ids,
        receiver_program.contrast_name,
    )
    if observed != expected:
        raise ContractError(
            "Receiver-program parent does not share the family-common fold lineage",
            code="family_common_parent_mismatch",
            field="receiver_program_training_artifact_id",
            remediation=(
                "Use the source-agnostic receiver program from the same receiver, "
                "contrast, and physical fold"
            ),
        )


def _frozen_interactions(
    receiver_family: ReceiverFamilyTrainingArtifact,
) -> tuple[FrozenFamilyInteraction, ...]:
    family_by_driver = {
        driver: family.family_id
        for family in receiver_family.family_basis.family_definitions
        for driver in family.driver_ids
    }
    gate_by_driver = dict(receiver_family.receptor_gates)
    eligible_by_driver = dict(
        zip(
            receiver_family.source_basis.driver_ids,
            receiver_family.source_basis.receptor_eligible.astype(bool).tolist(),
            strict=True,
        )
    )
    interactions = tuple(
        sorted(
            (
                FrozenFamilyInteraction(
                    interaction_id=interaction_id,
                    driver_id=driver_id,
                    family_id=family_by_driver[driver_id],
                    receptor_gate=gate_by_driver[driver_id],
                    receptor_eligible=bool(eligible_by_driver[driver_id]),
                )
                for interaction_id, driver_id in receiver_family.driver_by_interaction
            ),
            key=lambda item: item.interaction_id,
        )
    )
    if not interactions:
        raise ValueError("receiver-family artifact has no mapped interactions")
    return interactions


@validation_scope()
def fit_family_common_scoring_functional(
    receiver_family: ReceiverFamilyTrainingArtifact,
    incremental_functional: IncrementalDownstreamFunctional,
    sender_functional: ContrastCommonSenderFunctional,
    *,
    receiver_incremental_training_artifact_id: str,
    tuning_manifest_id: str,
    selected_penalty_id: str,
    autonomous_program_resource_id: str | None,
    receiver_program_artifact: ReceiverProgramTrainingArtifact | None = None,
    family_selection_threshold: float = 0.0,
    softmin_power: float = 4.0,
    epsilon: float = 1e-12,
) -> FamilyCommonScoringFunctional:
    """Compose one immutable family-first function without fitting held-out data."""

    if not isinstance(receiver_family, ReceiverFamilyTrainingArtifact):
        raise TypeError("receiver_family must be ReceiverFamilyTrainingArtifact")
    if not isinstance(incremental_functional, IncrementalDownstreamFunctional):
        raise TypeError(
            "incremental_functional must be IncrementalDownstreamFunctional"
        )
    if not isinstance(sender_functional, ContrastCommonSenderFunctional):
        raise TypeError("sender_functional must be ContrastCommonSenderFunctional")
    receiver_family._require_producer_owned()
    incremental_functional._require_intact()
    sender_functional._require_intact()
    _validate_parent_lineage(
        receiver_family,
        incremental_functional,
        sender_functional,
        autonomous_program_resource_id=autonomous_program_resource_id,
    )
    if receiver_program_artifact is not None:
        if not isinstance(
            receiver_program_artifact, ReceiverProgramTrainingArtifact
        ):
            raise TypeError(
                "receiver_program_artifact must be ReceiverProgramTrainingArtifact"
            )
        _validate_receiver_program_parent(
            receiver_family, receiver_program_artifact, sender_functional
        )
    lineage_names = {
        "receiver_incremental_training_artifact_id": (
            receiver_incremental_training_artifact_id
        ),
        "tuning_manifest_id": tuning_manifest_id,
        "selected_penalty_id": selected_penalty_id,
    }
    for field_name, value in lineage_names.items():
        _required_name(value, field_name=field_name)
    threshold = float(family_selection_threshold)
    power = float(softmin_power)
    shift = float(epsilon)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("family_selection_threshold must be finite in [0, 1)")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("softmin_power must be finite and positive")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("epsilon must be finite and positive")
    self = object.__new__(FamilyCommonScoringFunctional)
    values: dict[str, Any] = {
        "receiver_family": receiver_family,
        "receiver_program_artifact": receiver_program_artifact,
        "incremental_functional": incremental_functional,
        "sender_functional": sender_functional,
        "receiver": receiver_family.receiver,
        "contrast_name": sender_functional.contrast_name,
        "contrast_manifest_id": sender_functional.contrast_manifest_id,
        "fold_id": incremental_functional.fold_id,
        "context_ids": sender_functional.context_ids,
        "training_subject_ids": incremental_functional.training_subject_ids,
        "filter_universe_id": receiver_family.filter_universe_id,
        "family_ids": receiver_family.family_basis.family_ids,
        "active_family_ids": incremental_functional.family_ids,
        "interactions": _frozen_interactions(receiver_family),
        **lineage_names,
        "autonomous_program_resource_id": autonomous_program_resource_id,
        "receiver_program_reason_code": (
            "source_agnostic_receiver_program_parent_not_connected"
            if receiver_program_artifact is None
            else receiver_program_artifact.reason_code
        ),
        "incremental_reason_code": None,
        "family_selection_threshold": threshold,
        "softmin_power": power,
        "epsilon": shift,
        "score_version": _SCORE_VERSION,
        "certification_status": _CERTIFICATION_STATUS,
        "_producer_marker": _FUNCTIONAL_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "family_common_functional_id",
        stable_id(
            "family_common_scoring_functional",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    record_validation(self)
    return self


@validation_scope()
def mark_family_common_scoring_not_estimable(
    receiver_family: ReceiverFamilyTrainingArtifact,
    sender_functional: ContrastCommonSenderFunctional,
    *,
    fold_id: str,
    receiver_incremental_training_artifact_id: str,
    tuning_manifest_id: str,
    selected_penalty_id: str | None,
    autonomous_program_resource_id: str | None,
    reason_code: str,
    receiver_program_artifact: ReceiverProgramTrainingArtifact | None = None,
    family_selection_threshold: float = 0.0,
    softmin_power: float = 4.0,
    epsilon: float = 1e-12,
) -> FamilyCommonScoringFunctional:
    """Retain one planned receiver whose incremental parent is unavailable."""

    if not isinstance(receiver_family, ReceiverFamilyTrainingArtifact):
        raise TypeError("receiver_family must be ReceiverFamilyTrainingArtifact")
    if not isinstance(sender_functional, ContrastCommonSenderFunctional):
        raise TypeError("sender_functional must be ContrastCommonSenderFunctional")
    receiver_family._require_producer_owned()
    sender_functional._require_intact()
    if receiver_program_artifact is not None:
        if not isinstance(
            receiver_program_artifact, ReceiverProgramTrainingArtifact
        ):
            raise TypeError(
                "receiver_program_artifact must be ReceiverProgramTrainingArtifact"
            )
        _validate_receiver_program_parent(
            receiver_family, receiver_program_artifact, sender_functional
        )
    normalized_fold = _required_name(fold_id, field_name="fold_id")
    normalized_reason = _required_name(reason_code, field_name="reason_code")
    if receiver_family.fold_id != normalized_fold:
        raise ContractError(
            "Not-estimable family-common fold does not match receiver parent",
            code="family_common_parent_mismatch",
            field="fold_id",
            remediation="Use the planned receiver from the same physical fold",
        )
    if (
        receiver_family.training_subject_ids != sender_functional.training_subject_ids
        or receiver_family.filter_universe_id != sender_functional.filter_universe_id
    ):
        raise ContractError(
            "Not-estimable family-common parents do not share a training scope",
            code="family_common_parent_mismatch",
            field="parent_artifacts",
            remediation="Use receiver and sender parents from the same fold",
        )
    lineage_names = {
        "receiver_incremental_training_artifact_id": (
            _required_name(
                receiver_incremental_training_artifact_id,
                field_name="receiver_incremental_training_artifact_id",
            )
        ),
        "tuning_manifest_id": _required_name(
            tuning_manifest_id, field_name="tuning_manifest_id"
        ),
        "selected_penalty_id": (
            None
            if selected_penalty_id is None
            else _required_name(selected_penalty_id, field_name="selected_penalty_id")
        ),
    }
    threshold = float(family_selection_threshold)
    power = float(softmin_power)
    shift = float(epsilon)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("family_selection_threshold must be finite in [0, 1)")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("softmin_power must be finite and positive")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("epsilon must be finite and positive")
    self = object.__new__(FamilyCommonScoringFunctional)
    values: dict[str, Any] = {
        "receiver_family": receiver_family,
        "receiver_program_artifact": receiver_program_artifact,
        "incremental_functional": None,
        "sender_functional": sender_functional,
        "receiver": receiver_family.receiver,
        "contrast_name": sender_functional.contrast_name,
        "contrast_manifest_id": sender_functional.contrast_manifest_id,
        "fold_id": normalized_fold,
        "context_ids": sender_functional.context_ids,
        "training_subject_ids": receiver_family.training_subject_ids,
        "filter_universe_id": receiver_family.filter_universe_id,
        "family_ids": receiver_family.family_basis.family_ids,
        "active_family_ids": receiver_family.eligible_family_ids,
        "interactions": _frozen_interactions(receiver_family),
        **lineage_names,
        "autonomous_program_resource_id": autonomous_program_resource_id,
        "receiver_program_reason_code": (
            "source_agnostic_receiver_program_parent_not_connected"
            if receiver_program_artifact is None
            else receiver_program_artifact.reason_code
        ),
        "incremental_reason_code": normalized_reason,
        "family_selection_threshold": threshold,
        "softmin_power": power,
        "epsilon": shift,
        "score_version": _SCORE_VERSION,
        "certification_status": _CERTIFICATION_STATUS,
        "_producer_marker": _FUNCTIONAL_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "family_common_functional_id",
        stable_id(
            "family_common_scoring_functional",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    record_validation(self)
    return self


def _validate_incremental_application(
    functional: FamilyCommonScoringFunctional,
    application: IncrementalDownstreamApplication | None,
    *,
    heldout_reason_code: str | None,
) -> str | None:
    if functional.incremental_functional is None:
        if application is not None:
            raise ContractError(
                "Not-estimable functional cannot accept an incremental application",
                code="family_common_application_parent_mismatch",
                field="incremental_application_id",
                remediation="Use the typed not-estimable application path",
            )
        if heldout_reason_code is not None:
            return _required_name(heldout_reason_code, field_name="heldout_reason_code")
        return functional.incremental_reason_code
    if application is None:
        if heldout_reason_code is None:
            raise ContractError(
                "Missing held-out application requires an explicit reason",
                code="family_common_application_parent_mismatch",
                field="heldout_reason_code",
                remediation="Use the typed held-out not-estimable marker",
            )
        return _required_name(heldout_reason_code, field_name="heldout_reason_code")
    if heldout_reason_code is not None:
        raise ValueError(
            "observed incremental application cannot have a heldout reason"
        )
    if not isinstance(application, IncrementalDownstreamApplication):
        raise TypeError(
            "incremental_application must be IncrementalDownstreamApplication"
        )
    application._require_intact()
    if (
        application.incremental_functional_id
        != functional.incremental_functional.incremental_functional_id
        or application.family_ids != functional.active_family_ids
        or set(application.heldout_subject_ids).intersection(
            functional.training_subject_ids
        )
    ):
        raise ContractError(
            "Held-out incremental application does not match the common functional",
            code="family_common_application_parent_mismatch",
            field="incremental_application_id",
            remediation="Apply the exact frozen incremental parent to held-out rows",
        )
    reason: str | None = application.reason_code
    return reason


def _validated_sender_application(
    functional: FamilyCommonScoringFunctional,
    application: CommonSenderApplication,
) -> CommonSenderApplication:
    if not isinstance(application, CommonSenderApplication):
        raise TypeError("sender_application must be CommonSenderApplication")
    application._require_intact()
    if (
        application.functional.sender_functional_id
        != functional.sender_functional.sender_functional_id
    ):
        raise ContractError(
            "Sender application does not match the common scoring functional",
            code="family_common_application_parent_mismatch",
            field="sender_functional_id",
            remediation="Apply the exact contrast-common sender parent",
        )
    return application


def _validated_edge_evidence(
    functional: FamilyCommonScoringFunctional,
    incremental_application: IncrementalDownstreamApplication | None,
    edge_evidence: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(edge_evidence, pd.DataFrame):
        raise TypeError("edge_evidence must be a pandas DataFrame")
    if set(edge_evidence.columns) != set(FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS):
        raise ContractError(
            "Family-common edge evidence columns do not match the contract",
            code="invalid_family_common_edge_evidence",
            field="columns",
            remediation=f"Use exactly {list(FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS)}",
        )
    table = edge_evidence.loc[:, list(FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS)].copy()
    identifiers = (
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
        "ligand_contrast_gate_id",
    )
    for column in identifiers:
        table[column] = [
            _required_name(value, field_name=column) for value in table[column]
        ]
    numeric_columns = (
        "availability",
        "ligand_availability",
        "prior_quality",
        "subject_prevalence",
        "resource_evidence",
    )
    for column in numeric_columns:
        table[column] = [
            _unit_value(value, field_name=column) for value in table[column]
        ]
    try:
        table["ligand_contrast_gate"] = [
            _unit_value(value, field_name="ligand_contrast_gate")
            for value in table["ligand_contrast_gate"]
        ]
        table["ligand_contrast_gate_status"] = [
            SenderContrastSupportStatus(value).value
            for value in table["ligand_contrast_gate_status"]
        ]
        table["ligand_contrast_gate_reason_code"] = pd.Series(
            [
                None
                if value is None or value is pd.NA or pd.isna(value)
                else _required_name(
                    value, field_name="ligand_contrast_gate_reason_code"
                )
                for value in table["ligand_contrast_gate_reason_code"]
            ],
            index=table.index,
            dtype=object,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(
            "Family-common ligand contrast gate fields are invalid",
            code="invalid_family_common_edge_evidence",
            field="ligand_contrast_gate_status",
            remediation="Copy the exact frozen interaction gate into every row",
        ) from error
    key = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    if table.empty or table.duplicated(key).any():
        raise ContractError(
            "Family-common edge evidence keys must be non-empty and unique",
            code="invalid_family_common_edge_evidence",
            field="sample_id",
            remediation="Emit one explicit interaction row per sample and mode",
        )
    if set(table["receiver"]) != {functional.receiver}:
        raise ContractError(
            "Edge evidence receiver does not match the functional",
            code="family_common_application_parent_mismatch",
            field="receiver",
            remediation="Score only the receiver bound by the common functional",
        )
    expected_samples = (
        None
        if incremental_application is None
        else {
            sample_id: (subject_id, context_id)
            for sample_id, subject_id, context_id in zip(
                incremental_application.sample_ids,
                incremental_application.sample_subject_ids,
                incremental_application.sample_context_ids,
                strict=True,
            )
        }
    )
    observed_samples = {
        str(sample_id): (
            str(group["subject_id"].iloc[0]),
            str(group["context_id"].iloc[0]),
        )
        for sample_id, group in table.groupby("sample_id", observed=True, sort=False)
        if group["subject_id"].nunique() == 1 and group["context_id"].nunique() == 1
    }
    if len(observed_samples) != table["sample_id"].nunique():
        raise ContractError(
            "Each held-out sample must map to one subject and context",
            code="invalid_family_common_edge_evidence",
            field="sample_id",
            remediation="Restore immutable sample lineage before scoring",
        )
    if expected_samples is not None and observed_samples != expected_samples:
        raise ContractError(
            "Edge evidence does not exactly cover held-out sample lineage",
            code="family_common_application_parent_mismatch",
            field="sample_id",
            remediation="Join evidence to the exact held-out response row manifest",
        )
    if set(value[0] for value in observed_samples.values()).intersection(
        functional.training_subject_ids
    ):
        raise ContractError(
            "Family-common edge evidence overlaps training subjects",
            code="family_common_application_parent_mismatch",
            field="subject_id",
            remediation="Apply only to the physically held-out subject scope",
        )
    if set(table["context_id"]) != set(functional.context_ids):
        raise ContractError(
            "Edge evidence must contain every contrast context",
            code="invalid_family_common_edge_evidence",
            field="context_id",
            remediation="Apply one common function to the complete held-out contrast",
        )
    expected_interactions = set(functional.interaction_ids)
    expected_gates = dict(
        zip(
            functional.interaction_ids,
            interaction_ligand_contrast_gates(
                functional.sender_functional,
                tuple(
                    (functional.receiver, interaction_id)
                    for interaction_id in functional.interaction_ids
                ),
            ),
            strict=True,
        )
    )
    for row in table.itertuples(index=False):
        expected_gate = expected_gates.get(str(row.interaction_id))
        if expected_gate is None:
            raise ContractError(
                "Edge evidence contains an interaction outside the frozen parent",
                code="family_common_application_parent_mismatch",
                field="interaction_id",
                remediation="Use the exact frozen interaction set",
            )
        observed_gate = (
            _unit_value(
                row.ligand_contrast_gate,
                field_name="ligand_contrast_gate",
            ),
            row.ligand_contrast_gate_id,
            row.ligand_contrast_gate_status,
            row.ligand_contrast_gate_reason_code,
        )
        expected = (
            expected_gate.gate,
            expected_gate.gate_id,
            SenderContrastSupportStatus(expected_gate.status).value,
            expected_gate.reason_code,
        )
        if observed_gate != expected:
            raise ContractError(
                "Edge evidence ligand contrast gate does not match its sender parent",
                code="family_common_application_parent_mismatch",
                field="ligand_contrast_gate_id",
                remediation="Copy the exact frozen interaction gate into every row",
            )
    modes = tuple(sorted(set(table["mode"])))
    if modes != _RELEASED_MODES:
        raise ContractError(
            "Family-common evidence must contain the released scoring modes",
            code="invalid_family_common_edge_evidence",
            field="mode",
            remediation="Emit exact state and ecosystem rows for every sample edge",
        )
    coverage_samples = (
        observed_samples if expected_samples is None else expected_samples
    )
    interactions_by_sample_mode = {
        (str(sample_id), str(mode)): set(group["interaction_id"].astype(str))
        for (sample_id, mode), group in table.groupby(
            ["sample_id", "mode"], observed=True, sort=False
        )
    }
    for sample_id in coverage_samples:
        for mode in modes:
            observed_interactions = interactions_by_sample_mode.get(
                (sample_id, mode), set()
            )
            if observed_interactions != expected_interactions:
                raise ContractError(
                    "Every sample/mode must explicitly cover the frozen "
                    "interaction set",
                    code="invalid_family_common_edge_evidence",
                    field="interaction_id",
                    remediation=(
                        "Represent missing evidence as NA rows, not omitted edges"
                    ),
                )
    return table.sort_values(key, kind="stable", ignore_index=True)


def _validated_receiver_program_application(
    functional: FamilyCommonScoringFunctional,
    application: ReceiverProgramApplication | None,
    edge_evidence: pd.DataFrame,
) -> pd.DataFrame | None:
    parent = functional.receiver_program_artifact
    if parent is None:
        if application is not None:
            raise ContractError(
                "Functional without a receiver-program parent cannot accept one",
                code="family_common_application_parent_mismatch",
                field="receiver_program_application_id",
                remediation="Refit the family-common functional with that parent",
            )
        return None
    if application is None:
        raise ContractError(
            "Receiver-program parent requires an explicit held-out application",
            code="family_common_application_parent_mismatch",
            field="receiver_program_application_id",
            remediation=(
                "Apply or explicitly mark the source-agnostic receiver program "
                "on the exact held-out rows"
            ),
        )
    if not isinstance(application, ReceiverProgramApplication):
        raise TypeError(
            "receiver_program_application must be ReceiverProgramApplication"
        )
    application._require_intact()
    if (
        application.training_artifact.training_artifact_id
        != parent.training_artifact_id
    ):
        raise ContractError(
            "Receiver-program application does not match the frozen parent",
            code="family_common_application_parent_mismatch",
            field="receiver_program_application_id",
            remediation="Apply the exact receiver-program parent from this fold",
        )
    expected_samples = {
        str(sample_id): (
            str(group["subject_id"].iloc[0]),
            str(group["context_id"].iloc[0]),
        )
        for sample_id, group in edge_evidence.groupby(
            "sample_id", observed=True, sort=False
        )
    }
    observed_samples = {
        sample_id: (subject_id, context_id)
        for sample_id, subject_id, context_id in zip(
            application.sample_ids,
            application.sample_subject_ids,
            application.sample_context_ids,
            strict=True,
        )
    }
    if observed_samples != expected_samples:
        raise ContractError(
            "Receiver-program and edge evidence do not share exact held-out rows",
            code="family_common_application_parent_mismatch",
            field="receiver_program_application_id",
            remediation="Restore the exact sample/subject/context row manifest",
        )
    table = application.to_table()
    expected_keys = {
        (sample_id, family_id)
        for sample_id in expected_samples
        for family_id in functional.family_ids
    }
    observed_keys = set(
        zip(table["sample_id"], table["family_id"], strict=True)
    )
    if (
        observed_keys != expected_keys
        or set(table["receiver"]) != {functional.receiver}
        or table.duplicated(["sample_id", "family_id"]).any()
    ):
        raise ContractError(
            "Receiver-program application lacks exact sample-family coverage",
            code="family_common_application_parent_mismatch",
            field="family_id",
            remediation="Emit every frozen family once per held-out receiver sample",
        )
    return table


def _heldout_attribution_table(
    functional: FamilyCommonScoringFunctional,
    application: IncrementalDownstreamApplication | None,
    *,
    heldout_reason_code: str | None,
) -> pd.DataFrame:
    active_index = {
        family_id: index for index, family_id in enumerate(functional.active_family_ids)
    }
    rows: list[dict[str, object]] = []
    for family_id in functional.family_ids:
        index = active_index.get(family_id)
        if index is None:
            rows.append(
                {
                    "family_common_functional_id": (
                        functional.family_common_functional_id
                    ),
                    "family_id": family_id,
                    "family_coefficient": None,
                    "family_estimable": False,
                    "family_gain": None,
                    "raw_family_gain": None,
                    "family_selected": False,
                    "family_selection_frequency": None,
                    "selection_status": "structural_zero",
                    "reason_code": "receptor_family_ineligible",
                }
            )
            continue
        incremental = functional.incremental_functional
        coefficient: float | None
        gain: float | None
        raw_gain: float | None
        selected: bool | None
        if incremental is None:
            coefficient = None
            estimable = False
            gain = None
            raw_gain = None
            selected = None
            selection_status = "not_estimable"
            reason = heldout_reason_code or functional.incremental_reason_code
        else:
            coefficient = float(incremental.family_coefficients[index])
            estimable = bool(incremental.family_estimable[index])
            if application is None or application.status != "observed":
                gain = None
                raw_gain = None
                selected = None
                selection_status = "not_estimable"
                reason = (
                    heldout_reason_code
                    if application is None
                    else application.reason_code
                    or "incremental_downstream_not_estimable"
                )
            elif not estimable:
                gain = None
                raw_gain = None
                selected = None
                selection_status = "not_estimable"
                reason = incremental.family_reason_codes[index]
            else:
                assert coefficient is not None
                gain = float(application.family_gains[index])
                raw_gain = float(application.raw_family_gains[index])
                selected = (
                    coefficient > 0.0 and gain > functional.family_selection_threshold
                )
                selection_status = "selected" if selected else "not_selected"
                reason = None if selected else "family_below_selection_threshold"
        rows.append(
            {
                "family_common_functional_id": functional.family_common_functional_id,
                "family_id": family_id,
                "family_coefficient": coefficient,
                "family_estimable": estimable,
                "family_gain": gain,
                "raw_family_gain": raw_gain,
                "family_selected": selected,
                "family_selection_frequency": None,
                "selection_status": selection_status,
                "reason_code": reason,
            }
        )
    return pd.DataFrame(rows, columns=FAMILY_HELDOUT_ATTRIBUTION_COLUMNS)


def _subject_differential_table(
    functional: FamilyCommonScoringFunctional,
    application: IncrementalDownstreamApplication | None,
    *,
    heldout_subject_ids: tuple[str, ...],
    heldout_reason_code: str | None,
) -> pd.DataFrame:
    active_index = {
        family_id: index for index, family_id in enumerate(functional.active_family_ids)
    }
    application_subject_index = (
        {}
        if application is None
        else {subject: index for index, subject in enumerate(application.subject_ids)}
    )
    incremental = functional.incremental_functional
    rows: list[dict[str, object]] = []
    for subject_id in heldout_subject_ids:
        for family_id in functional.family_ids:
            family_index = active_index.get(family_id)
            subject_index = application_subject_index.get(subject_id)
            if family_index is None:
                null_loss: float | None = None
                family_loss: float | None = None
                raw_effect: float | None = 0.0
                bounded_gain: float | None = 0.0
                status = "structural_zero"
                reason: str | None = "receptor_family_ineligible"
            elif (
                incremental is None
                or application is None
                or application.status != "observed"
                or subject_index is None
                or not bool(incremental.family_estimable[family_index])
            ):
                null_loss = None
                family_loss = None
                raw_effect = None
                bounded_gain = None
                status = "not_estimable"
                reason = (
                    heldout_reason_code
                    or (None if application is None else application.reason_code)
                    or "subject_family_incremental_not_estimable"
                )
            else:
                null_loss = float(application.subject_null_losses[subject_index])
                family_loss = float(
                    application.subject_family_losses[subject_index, family_index]
                )
                if null_loss <= incremental.null_loss_floor:
                    raw_effect = 0.0
                    bounded_gain = 0.0
                    status = "structural_zero"
                    reason = "zero_receiver_contrast_structural_zero_v1"
                else:
                    raw_effect = (null_loss - family_loss) / null_loss
                    bounded_gain = (
                        0.0
                        if raw_effect <= incremental.null_loss_floor
                        else float(np.clip(raw_effect, 0.0, 1.0))
                    )
                    status = "observed"
                    reason = None
            rows.append(
                {
                    "family_common_functional_id": (
                        functional.family_common_functional_id
                    ),
                    "subject_id": subject_id,
                    "family_id": family_id,
                    "null_loss": null_loss,
                    "family_loss": family_loss,
                    "differential_effect": raw_effect,
                    "bounded_incremental_gain": bounded_gain,
                    "status": status,
                    "reason_code": reason,
                }
            )
    return pd.DataFrame(rows, columns=FAMILY_SUBJECT_DIFFERENTIAL_COLUMNS)


def _normalized_entropy(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 0.0
    return float(
        -sum(weight * math.log(weight) for weight in weights if weight > 0)
        / math.log(len(weights))
    )


def _score_tables(
    functional: FamilyCommonScoringFunctional,
    edge_evidence: pd.DataFrame,
    attribution: pd.DataFrame,
    subject_differential: pd.DataFrame,
    receiver_program: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = pd.DataFrame([item.to_dict() for item in functional.interactions])
    edges = edge_evidence.merge(
        membership,
        how="inner",
        on="interaction_id",
        validate="many_to_one",
        sort=False,
    )
    attribution_by_family = attribution.set_index("family_id", drop=False)
    differential_by_subject_family = subject_differential.set_index(
        ["subject_id", "family_id"], drop=False
    )
    program_by_sample_family = (
        None
        if receiver_program is None
        else receiver_program.set_index(["sample_id", "family_id"], drop=False)
    )
    family_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    group_keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "mode",
        "family_id",
    ]
    for group_key, group in edges.groupby(group_keys, observed=True, sort=True):
        sample_id, subject_id, context_id, receiver, mode, family_id = group_key
        family_attribution = cast(pd.Series, attribution_by_family.loc[str(family_id)])
        subject_effect = cast(
            pd.Series,
            differential_by_subject_family.loc[(str(subject_id), str(family_id))],
        )
        program_row = (
            None
            if program_by_sample_family is None
            else cast(
                pd.Series,
                program_by_sample_family.loc[(str(sample_id), str(family_id))],
            )
        )
        if program_row is None:
            receiver_program_score: float | None = None
            receiver_program_status = "not_estimable"
            receiver_program_reason = functional.receiver_program_reason_code
        else:
            raw_program_score = program_row["receiver_program_score"]
            receiver_program_score = (
                None
                if pd.isna(raw_program_score)
                else _unit_value(
                    raw_program_score, field_name="receiver_program_score"
                )
            )
            receiver_program_status = str(program_row["status"])
            raw_program_reason = program_row["reason_code"]
            receiver_program_reason = (
                None if pd.isna(raw_program_reason) else str(raw_program_reason)
            )
        eligible = group["receptor_eligible"].astype(bool)
        gate_supported = group["ligand_contrast_gate_status"].eq(
            SenderContrastSupportStatus.SUPPORTED.value
        )
        gate_not_estimable = group["ligand_contrast_gate_status"].eq(
            SenderContrastSupportStatus.NOT_ESTIMABLE.value
        )
        eligible_supported = eligible & gate_supported
        eligible_not_estimable = eligible & gate_not_estimable
        family_receptor_eligible = bool(eligible.any())
        family_gate_supported = bool(eligible_supported.any())
        family_gate_not_estimable = bool(eligible_not_estimable.any())
        supported_interaction_count = int(eligible_supported.sum())
        not_estimable_interaction_count = int(eligible_not_estimable.sum())
        if not family_receptor_eligible:
            family_gate_status = "not_applicable_receptor_ineligible"
        elif family_gate_supported:
            family_gate_status = SenderContrastSupportStatus.SUPPORTED.value
        elif family_gate_not_estimable:
            family_gate_status = SenderContrastSupportStatus.NOT_ESTIMABLE.value
        else:
            family_gate_status = SenderContrastSupportStatus.UNSUPPORTED.value
        availability_values = [
            _unit_value(value, field_name="availability")
            for value in group.loc[eligible_supported, "availability"]
        ]
        if not family_receptor_eligible:
            family_availability: float | None = 0.0
        elif not family_gate_supported:
            family_availability = (
                None if family_gate_not_estimable else 0.0
            )
        elif any(value is None for value in availability_values):
            family_availability = None
        else:
            family_availability = max(cast(list[float], availability_values))
        family_gain = _unit_value(
            subject_effect["bounded_incremental_gain"],
            field_name="bounded_incremental_gain",
        )
        differential_effect_raw = subject_effect["differential_effect"]
        differential_effect = (
            None
            if pd.isna(differential_effect_raw)
            else float(cast(Any, differential_effect_raw))
        )
        selected_raw = family_attribution["family_selected"]
        family_selected = (
            None if pd.isna(selected_raw) else bool(cast(Any, selected_raw))
        )
        if not family_receptor_eligible:
            core: float | None = 0.0
            core_status = "structural_zero"
            core_reason: str | None = "receptor_family_ineligible"
        elif not family_gate_supported and family_gate_not_estimable:
            core = None
            core_status = "not_estimable"
            core_reason = "ligand_contrast_not_estimable"
        elif not family_gate_supported:
            core = 0.0
            core_status = "structural_zero"
            core_reason = "ligand_contrast_not_supported"
        elif family_selected is False:
            core = 0.0
            core_status = "structural_zero"
            core_reason = "family_not_selected"
        elif family_selected is None:
            core = None
            core_status = "not_estimable"
            core_reason = cast(str | None, family_attribution["reason_code"])
        elif family_availability == 0.0:
            core = 0.0
            core_status = "structural_zero"
            core_reason = "family_availability_zero"
        elif family_gain == 0.0:
            core = 0.0
            core_status = "structural_zero"
            core_reason = "incremental_downstream_gain_zero"
        elif family_availability is None:
            core = None
            core_status = "not_estimable"
            core_reason = "family_availability_missing"
        elif family_gain is None:
            core = None
            core_status = "not_estimable"
            core_reason = cast(str | None, subject_effect["reason_code"])
        else:
            core = pair_softmin(
                family_availability,
                family_gain,
                power=functional.softmin_power,
                epsilon=functional.epsilon,
            )
            core_status = "ok"
            core_reason = None

        group_rows = list(group.itertuples(index=False))
        gate_statuses = [
            SenderContrastSupportStatus(str(row.ligand_contrast_gate_status))
            for row in group_rows
        ]
        evidence_scores: list[float | None] = []
        supported_indices: list[int] = []
        eligible_gate_ne = False
        for index, (row, gate_status) in enumerate(
            zip(group_rows, gate_statuses, strict=True)
        ):
            if not bool(row.receptor_eligible):
                evidence_scores.append(0.0)
                continue
            if gate_status is SenderContrastSupportStatus.UNSUPPORTED:
                evidence_scores.append(0.0)
                continue
            if gate_status is SenderContrastSupportStatus.NOT_ESTIMABLE:
                evidence_scores.append(None)
                eligible_gate_ne = True
                continue
            supported_indices.append(index)
            evidence_components = (
                _unit_value(row.availability, field_name="availability"),
                float(bool(row.receptor_eligible)),
                _unit_value(
                    row.ligand_contrast_gate,
                    field_name="ligand_contrast_gate",
                ),
                _unit_value(row.receptor_gate, field_name="receptor_gate"),
                _unit_value(row.ligand_availability, field_name="ligand_availability"),
                _unit_value(row.prior_quality, field_name="prior_quality"),
                _unit_value(row.subject_prevalence, field_name="subject_prevalence"),
                _unit_value(row.resource_evidence, field_name="resource_evidence"),
            )
            if any(value is None for value in evidence_components):
                evidence_scores.append(None)
            else:
                evidence_scores.append(
                    math.prod(cast(tuple[float, ...], evidence_components))
                )
        weights: list[float | None] = [
            None
            if bool(row.receptor_eligible)
            and gate_status is not SenderContrastSupportStatus.UNSUPPORTED
            else 0.0
            for row, gate_status in zip(group_rows, gate_statuses, strict=True)
        ]
        supported_scores = [evidence_scores[index] for index in supported_indices]
        if not supported_indices:
            entropy: float | None = None
            identifiability = "unresolved" if eligible_gate_ne else "resolved"
            allocation_reason: str | None = (
                "ligand_contrast_not_estimable"
                if eligible_gate_ne
                else "ligand_contrast_not_supported"
            )
        elif eligible_gate_ne:
            entropy = None
            identifiability = "unresolved"
            allocation_reason = "ligand_contrast_not_estimable"
        elif any(value is None for value in supported_scores):
            entropy = None
            identifiability = "unresolved"
            allocation_reason = "incomplete_member_evidence"
        else:
            numeric_scores = cast(list[float], supported_scores)
            total = sum(numeric_scores)
            if total <= 0:
                entropy = None
                identifiability = "unresolved"
                allocation_reason = "no_positive_member_evidence"
            else:
                normalized_weights = [value / total for value in numeric_scores]
                for index, normalized_weight in zip(
                    supported_indices, normalized_weights, strict=True
                ):
                    weights[index] = normalized_weight
                entropy = _normalized_entropy(normalized_weights)
                identifiability = "resolved"
                allocation_reason = None

        family_rows.append(
            {
                "family_common_functional_id": functional.family_common_functional_id,
                "sample_id": sample_id,
                "subject_id": subject_id,
                "context_id": context_id,
                "receiver": receiver,
                "family_id": family_id,
                "mode": mode,
                "availability_score": family_availability,
                "family_availability": family_availability,
                "receptor_eligible": family_receptor_eligible,
                "ligand_contrast_gate_status": family_gate_status,
                "ligand_contrast_supported_interaction_count": (
                    supported_interaction_count
                ),
                "ligand_contrast_not_estimable_interaction_count": (
                    not_estimable_interaction_count
                ),
                "receiver_program_score": receiver_program_score,
                "receiver_program_status": receiver_program_status,
                "receiver_program_reason_code": receiver_program_reason,
                "incremental_downstream_gain": family_gain,
                "differential_effect": differential_effect,
                "family_coefficient": family_attribution["family_coefficient"],
                "family_selected": family_attribution["family_selected"],
                "within_family_entropy": entropy,
                "lr_identifiability_status": identifiability,
                "family_core_strength": core,
                "integrated_lr_score": core,
                "status": core_status,
                "reason_code": core_reason,
                "score_version": functional.score_version,
            }
        )
        for row, gate_status, evidence_score, member_weight in zip(
            group_rows, gate_statuses, evidence_scores, weights, strict=True
        ):
            unresolved: float | None
            status: str
            reason: str | None
            if not bool(row.receptor_eligible):
                unresolved = 0.0
                status = "structural_zero"
                reason = "receptor_interaction_ineligible"
            elif gate_status is SenderContrastSupportStatus.UNSUPPORTED:
                unresolved = 0.0
                status = "structural_zero"
                reason = "ligand_contrast_not_supported"
            elif core == 0.0:
                unresolved = 0.0
                status = "structural_zero"
                reason = core_reason
            elif gate_status is SenderContrastSupportStatus.NOT_ESTIMABLE:
                unresolved = None
                status = "not_estimable"
                reason = str(
                    row.ligand_contrast_gate_reason_code
                    or "ligand_contrast_not_estimable"
                )
            elif core is None:
                unresolved = None
                status = "not_estimable"
                reason = core_reason
            elif member_weight is None:
                unresolved = None
                status = "not_estimable"
                reason = allocation_reason
            else:
                unresolved = core * member_weight
                status = "ok"
                reason = None
            member_rows.append(
                {
                    "family_common_functional_id": (
                        functional.family_common_functional_id
                    ),
                    "sample_id": row.sample_id,
                    "subject_id": row.subject_id,
                    "context_id": row.context_id,
                    "receiver": row.receiver,
                    "family_id": row.family_id,
                    "driver_id": row.driver_id,
                    "interaction_id": row.interaction_id,
                    "mode": row.mode,
                    "availability": row.availability,
                    "receptor_gate": row.receptor_gate,
                    "receptor_eligible": bool(row.receptor_eligible),
                    "ligand_contrast_gate": row.ligand_contrast_gate,
                    "ligand_contrast_gate_id": row.ligand_contrast_gate_id,
                    "ligand_contrast_gate_status": row.ligand_contrast_gate_status,
                    "ligand_contrast_gate_reason_code": (
                        row.ligand_contrast_gate_reason_code
                    ),
                    "ligand_availability": row.ligand_availability,
                    "prior_quality": row.prior_quality,
                    "subject_prevalence": row.subject_prevalence,
                    "resource_evidence": row.resource_evidence,
                    "member_evidence_score": evidence_score,
                    "within_family_lr_weight": member_weight,
                    "within_family_entropy": entropy,
                    "lr_identifiability_status": identifiability,
                    "family_core_strength": core,
                    "sender_unresolved_strength": unresolved,
                    "status": status,
                    "reason_code": reason,
                    "score_version": functional.score_version,
                }
            )
    family_table = pd.DataFrame(family_rows, columns=FAMILY_COMMON_SCORE_COLUMNS)
    member_table = pd.DataFrame(member_rows, columns=FAMILY_MEMBER_SCORE_COLUMNS)
    return (
        family_table.sort_values(
            ["sample_id", "context_id", "receiver", "family_id", "mode"],
            kind="stable",
            ignore_index=True,
        ),
        member_table.sort_values(
            [
                "sample_id",
                "context_id",
                "receiver",
                "family_id",
                "interaction_id",
                "mode",
            ],
            kind="stable",
            ignore_index=True,
        ),
    )


def _sender_score_table(
    functional: FamilyCommonScoringFunctional,
    member_scores: pd.DataFrame,
    sender_application: CommonSenderApplication,
) -> pd.DataFrame:
    assignment = sender_application.table.copy(deep=True)
    member_keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
    ]
    interaction_ids = set(functional.interaction_ids)
    candidate_rows = [
        {
            "receiver": prior.receiver,
            "interaction_id": prior.interaction_id,
            "sender": prior.sender,
        }
        for prior in functional.sender_functional.candidate_priors
        if prior.receiver == functional.receiver
        and prior.interaction_id in interaction_ids
    ]
    candidates = pd.DataFrame(
        candidate_rows, columns=("receiver", "interaction_id", "sender")
    )
    expected = member_scores.merge(
        candidates,
        how="inner",
        on=["receiver", "interaction_id"],
        validate="many_to_many",
        sort=False,
    )
    assignment_values = assignment.loc[
        :,
        [
            *member_keys,
            "sender",
            "assignment_weight",
            "status",
            "reason_code",
        ],
    ].rename(
        columns={
            "status": "sender_assignment_status",
            "reason_code": "sender_assignment_reason_code",
        }
    )
    merged = expected.merge(
        assignment_values,
        how="left",
        on=[*member_keys, "sender"],
        validate="many_to_one",
        sort=False,
    )
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        unresolved = _unit_value(
            row.sender_unresolved_strength,
            field_name="sender_unresolved_strength",
        )
        assignment_weight = _unit_value(
            row.assignment_weight, field_name="assignment_weight"
        )
        if unresolved == 0.0:
            resolved: float | None = 0.0
            status = "structural_zero"
            reason = row.reason_code
        elif unresolved is None:
            resolved = None
            status = "not_estimable"
            reason = row.reason_code
        elif assignment_weight is None:
            resolved = None
            status = "not_estimable"
            reason = (
                "sender_application_group_missing"
                if pd.isna(row.sender_assignment_status)
                else row.sender_assignment_reason_code
            )
        else:
            resolved = unresolved * assignment_weight
            status = "ok"
            reason = None
        rows.append(
            {
                "family_common_functional_id": functional.family_common_functional_id,
                "sender_functional_id": (
                    sender_application.functional.sender_functional_id
                ),
                "sample_id": row.sample_id,
                "subject_id": row.subject_id,
                "context_id": row.context_id,
                "receiver": row.receiver,
                "family_id": row.family_id,
                "driver_id": row.driver_id,
                "interaction_id": row.interaction_id,
                "mode": row.mode,
                "sender": row.sender,
                "sender_unresolved_strength": unresolved,
                "assignment_weight": assignment_weight,
                "sender_resolved_strength": resolved,
                "status": status,
                "reason_code": reason,
                "score_version": functional.score_version,
            }
        )
    result = pd.DataFrame(rows, columns=FAMILY_SENDER_SCORE_COLUMNS)
    if result.empty:
        return result
    group_keys = [*member_keys, "mode"]
    for _, group in result.groupby(group_keys, observed=True, sort=False):
        unresolved_values = group["sender_unresolved_strength"].dropna().unique()
        if len(unresolved_values) != 1:
            continue
        unresolved = float(unresolved_values[0])
        resolved_series = group["sender_resolved_strength"]
        if resolved_series.notna().all() and not math.isclose(
            float(resolved_series.sum()), unresolved, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise RuntimeError("sender resolution failed family-member conservation")
    return result.sort_values([*group_keys, "sender"], kind="stable", ignore_index=True)


@dataclass(frozen=True, slots=True, init=False)
class FamilyCommonScoringApplication:
    """Producer-owned held-out family/member/sender score collection."""

    functional: FamilyCommonScoringFunctional
    receiver_program_application_id: str | None
    incremental_application_id: str | None
    heldout_reason_code: str | None
    edge_evidence_digest: str
    sender_application_digest: str
    heldout_subject_ids: tuple[str, ...]
    family_attribution_digest: str
    subject_differential_digest: str
    family_scores_digest: str
    member_scores_digest: str
    sender_scores_digest: str
    table_row_counts: tuple[int, int, int, int, int]
    application_id: str
    _family_attribution: pd.DataFrame
    _subject_differential: pd.DataFrame
    _family_scores: pd.DataFrame
    _member_scores: pd.DataFrame
    _sender_scores: pd.DataFrame
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FamilyCommonScoringApplication is producer-owned; use "
            "apply_family_common_scoring_functional()"
        )

    @property
    def is_oof_certified(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "certification_status": self.functional.certification_status,
            "edge_evidence_digest": self.edge_evidence_digest,
            "family_attribution_digest": self.family_attribution_digest,
            "family_common_functional_id": self.functional.family_common_functional_id,
            "family_scores_digest": self.family_scores_digest,
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "heldout_reason_code": self.heldout_reason_code,
            "incremental_application_id": self.incremental_application_id,
            "receiver_program_application_id": (
                self.receiver_program_application_id
            ),
            "member_scores_digest": self.member_scores_digest,
            "sender_application_digest": self.sender_application_digest,
            "sender_scores_digest": self.sender_scores_digest,
            "subject_differential_digest": self.subject_differential_digest,
            "table_row_counts": list(self.table_row_counts),
        }

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            self.functional._require_intact()
            observed_digests = (
                _table_digest(
                    "family_heldout_attribution",
                    self._family_attribution,
                    FAMILY_HELDOUT_ATTRIBUTION_COLUMNS,
                ),
                _table_digest(
                    "family_subject_differential",
                    self._subject_differential,
                    FAMILY_SUBJECT_DIFFERENTIAL_COLUMNS,
                ),
                _table_digest(
                    "family_common_scores",
                    self._family_scores,
                    FAMILY_COMMON_SCORE_COLUMNS,
                ),
                _table_digest(
                    "family_member_scores",
                    self._member_scores,
                    FAMILY_MEMBER_SCORE_COLUMNS,
                ),
                _table_digest(
                    "family_sender_scores",
                    self._sender_scores,
                    FAMILY_SENDER_SCORE_COLUMNS,
                ),
            )
            valid = (
                self._producer_marker == _APPLICATION_PRODUCER
                and observed_digests
                == (
                    self.family_attribution_digest,
                    self.subject_differential_digest,
                    self.family_scores_digest,
                    self.member_scores_digest,
                    self.sender_scores_digest,
                )
                and self.table_row_counts
                == tuple(
                    len(table)
                    for table in (
                        self._family_attribution,
                        self._subject_differential,
                        self._family_scores,
                        self._member_scores,
                        self._sender_scores,
                    )
                )
                and not set(self.heldout_subject_ids).intersection(
                    self.functional.training_subject_ids
                )
                and isinstance(self.edge_evidence_digest, str)
                and bool(self.edge_evidence_digest)
                and isinstance(self.sender_application_digest, str)
                and bool(self.sender_application_digest)
                and (
                    self.incremental_application_id is not None
                    or bool(self.heldout_reason_code)
                )
                and (
                    (
                        self.functional.receiver_program_artifact is None
                        and self.receiver_program_application_id is None
                    )
                    or (
                        self.functional.receiver_program_artifact is not None
                        and bool(self.receiver_program_application_id)
                    )
                )
                and not self.is_oof_certified
                and stable_id(
                    "family_common_scoring_application",
                    self._identity_payload(),
                    schema_version="2",
                )
                == self.application_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Family-common application failed integrity validation",
                code="family_common_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact common functional to held-out parents",
            ) from error
        if not valid:
            raise ContractError(
                "Family-common application failed integrity validation",
                code="family_common_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact common functional to held-out parents",
            )
        record_validation(self)

    @property
    def family_attribution(self) -> pd.DataFrame:
        self._require_intact()
        return self._family_attribution.copy(deep=True)

    @property
    def family_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._family_scores.copy(deep=True)

    @property
    def subject_differential(self) -> pd.DataFrame:
        self._require_intact()
        return self._subject_differential.copy(deep=True)

    @property
    def member_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._member_scores.copy(deep=True)

    @property
    def sender_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._sender_scores.copy(deep=True)

    def _validated_tables(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return internal tables after one complete integrity check."""

        self._require_intact()
        return (
            self._family_attribution,
            self._subject_differential,
            self._family_scores,
            self._member_scores,
            self._sender_scores,
        )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "application_id": self.application_id,
            **self._identity_payload(),
            "score_version": self.functional.score_version,
            "selection_frequency_status": _SELECTION_FREQUENCY_REASON,
            "is_oof_certified": self.is_oof_certified,
        }


@validation_scope()
def apply_family_common_scoring_functional(
    functional: FamilyCommonScoringFunctional,
    incremental_application: IncrementalDownstreamApplication | None,
    edge_evidence: pd.DataFrame,
    sender_application: CommonSenderApplication,
    *,
    receiver_program_application: ReceiverProgramApplication | None = None,
    heldout_reason_code: str | None = None,
) -> FamilyCommonScoringApplication:
    """Apply one common function to held-out family and sender evidence."""

    if not isinstance(functional, FamilyCommonScoringFunctional):
        raise TypeError("functional must be FamilyCommonScoringFunctional")
    functional._require_intact()
    resolved_heldout_reason = _validate_incremental_application(
        functional,
        incremental_application,
        heldout_reason_code=heldout_reason_code,
    )
    sender = _validated_sender_application(functional, sender_application)
    edges = _validated_edge_evidence(functional, incremental_application, edge_evidence)
    receiver_program = _validated_receiver_program_application(
        functional, receiver_program_application, edges
    )
    sender_samples = {
        str(sample_id): (
            str(group["subject_id"].iloc[0]),
            str(group["context_id"].iloc[0]),
        )
        for sample_id, group in sender.table.groupby(
            "sample_id", observed=True, sort=False
        )
        if group["subject_id"].nunique() == 1 and group["context_id"].nunique() == 1
    }
    expected_samples = {
        str(sample_id): (
            str(group["subject_id"].iloc[0]),
            str(group["context_id"].iloc[0]),
        )
        for sample_id, group in edges.groupby("sample_id", observed=True, sort=False)
    }
    if any(
        sample_id not in expected_samples or expected_samples[sample_id] != lineage
        for sample_id, lineage in sender_samples.items()
    ):
        raise ContractError(
            "Sender and incremental applications do not share held-out rows",
            code="family_common_application_parent_mismatch",
            field="sender_application",
            remediation="Apply both frozen parents to the exact held-out sample scope",
        )
    attribution = _heldout_attribution_table(
        functional,
        incremental_application,
        heldout_reason_code=resolved_heldout_reason,
    )
    heldout_subject_ids = tuple(sorted(set(edges["subject_id"])))
    subject_differential = _subject_differential_table(
        functional,
        incremental_application,
        heldout_subject_ids=heldout_subject_ids,
        heldout_reason_code=resolved_heldout_reason,
    )
    family_scores, member_scores = _score_tables(
        functional,
        edges,
        attribution,
        subject_differential,
        receiver_program,
    )
    sender_scores = _sender_score_table(functional, member_scores, sender)
    edge_evidence_digest = family_common_edge_evidence_digest(edges)
    sender_application_digest = family_common_sender_application_digest(sender)
    family_attribution_digest = _table_digest(
        "family_heldout_attribution",
        attribution,
        FAMILY_HELDOUT_ATTRIBUTION_COLUMNS,
    )
    subject_differential_digest = _table_digest(
        "family_subject_differential",
        subject_differential,
        FAMILY_SUBJECT_DIFFERENTIAL_COLUMNS,
    )
    family_scores_digest = _table_digest(
        "family_common_scores", family_scores, FAMILY_COMMON_SCORE_COLUMNS
    )
    member_scores_digest = _table_digest(
        "family_member_scores", member_scores, FAMILY_MEMBER_SCORE_COLUMNS
    )
    sender_scores_digest = _table_digest(
        "family_sender_scores", sender_scores, FAMILY_SENDER_SCORE_COLUMNS
    )
    self = object.__new__(FamilyCommonScoringApplication)
    values: dict[str, Any] = {
        "functional": functional,
        "receiver_program_application_id": (
            None
            if receiver_program_application is None
            else receiver_program_application.application_id
        ),
        "incremental_application_id": (
            None
            if incremental_application is None
            else incremental_application.application_id
        ),
        "heldout_reason_code": resolved_heldout_reason,
        "edge_evidence_digest": edge_evidence_digest,
        "sender_application_digest": sender_application_digest,
        "heldout_subject_ids": heldout_subject_ids,
        "family_attribution_digest": family_attribution_digest,
        "subject_differential_digest": subject_differential_digest,
        "family_scores_digest": family_scores_digest,
        "member_scores_digest": member_scores_digest,
        "sender_scores_digest": sender_scores_digest,
        "table_row_counts": (
            len(attribution),
            len(subject_differential),
            len(family_scores),
            len(member_scores),
            len(sender_scores),
        ),
        "_family_attribution": attribution.copy(deep=True),
        "_subject_differential": subject_differential.copy(deep=True),
        "_family_scores": family_scores.copy(deep=True),
        "_member_scores": member_scores.copy(deep=True),
        "_sender_scores": sender_scores.copy(deep=True),
        "_producer_marker": _APPLICATION_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "application_id",
        stable_id(
            "family_common_scoring_application",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    record_validation(self)
    return self


def mark_family_common_scoring_application_not_estimable(
    functional: FamilyCommonScoringFunctional,
    edge_evidence: pd.DataFrame,
    sender_application: CommonSenderApplication,
    *,
    heldout_reason_code: str,
    receiver_program_application: ReceiverProgramApplication | None = None,
) -> FamilyCommonScoringApplication:
    """Emit exact held-out coverage for a planned unavailable receiver model."""

    if not isinstance(functional, FamilyCommonScoringFunctional):
        raise TypeError("functional must be FamilyCommonScoringFunctional")
    functional._require_intact()
    normalized_reason = _required_name(
        heldout_reason_code, field_name="heldout_reason_code"
    )
    return apply_family_common_scoring_functional(
        functional,
        None,
        edge_evidence,
        sender_application,
        receiver_program_application=receiver_program_application,
        heldout_reason_code=normalized_reason,
    )


__all__ = [
    "FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS",
    "FAMILY_COMMON_SCORE_COLUMNS",
    "FAMILY_COMMON_SCORE_VERSION",
    "FAMILY_HELDOUT_ATTRIBUTION_COLUMNS",
    "FAMILY_MEMBER_SCORE_COLUMNS",
    "FAMILY_SENDER_SCORE_COLUMNS",
    "FAMILY_SUBJECT_DIFFERENTIAL_COLUMNS",
    "FamilyCommonScoringApplication",
    "FamilyCommonScoringFunctional",
    "FrozenFamilyInteraction",
    "apply_family_common_scoring_functional",
    "family_common_edge_evidence_digest",
    "family_common_sender_application_digest",
    "fit_family_common_scoring_functional",
    "mark_family_common_scoring_application_not_estimable",
    "mark_family_common_scoring_not_estimable",
]
