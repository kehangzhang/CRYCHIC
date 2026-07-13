"""Training-only receptor gates and strict receiver-family bases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crychic.availability import (
    BatchAvailability,
    InteractionFilterApplication,
)
from crychic.core import stable_id
from crychic.resources import TargetPrior

from .basis import build_gated_target_basis
from .contracts import FamilyFirstBasis, GatedTargetBasis, ReceptorGatePolicy
from .families import cluster_driver_families
from .family_first import build_family_first_basis

_PRODUCER_MARKER = "crychic.receiver_family_training.v1"
_TRAINING_STATUS = "training_only_partial_receiver_family_v1"
_POOLING_METHOD = "condition_blind_sample_max_then_mean_v1"
_COMPLETED_STAGES = (
    "condition_blind_receptor_gate",
    "strict_family_partition",
    "strict_medoid_family_basis",
)
_REMAINING_STAGES = (
    "response_precision",
    "family_attribution",
    "attribution_tuning",
    "incremental_downstream",
    "common_scoring_functional",
)
_RECEPTOR_COLUMNS = {
    "sample_id",
    "subject_id",
    "receiver",
    "interaction_id",
    "receptor_availability",
}


def _nonempty_name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _driver_mapping(
    values: Mapping[str, str], prior: TargetPrior
) -> tuple[tuple[str, str], ...]:
    mapping: list[tuple[str, str]] = []
    known_drivers = set(prior.driver_ids)
    for raw_interaction, raw_driver in values.items():
        interaction = _nonempty_name(raw_interaction, field_name="interaction_id")
        driver = _nonempty_name(raw_driver, field_name="driver_id")
        if driver not in known_drivers:
            raise ValueError(
                f"driver_by_interaction contains unknown prior driver {driver!r}"
            )
        mapping.append((interaction, driver))
    return tuple(sorted(mapping))


def _pooled_receptor_gates(
    availability: BatchAvailability,
    *,
    receiver: str,
    prior: TargetPrior,
    driver_by_interaction: tuple[tuple[str, str], ...],
) -> tuple[tuple[tuple[str, float], ...], str]:
    table = availability.sample_interactions
    missing = _RECEPTOR_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(
            f"training availability is missing receptor columns: {sorted(missing)}"
        )
    observed_subjects = set(table["subject_id"].dropna().astype(str))
    unknown_subjects = observed_subjects.difference(
        availability.application_subject_ids
    )
    if unknown_subjects:
        raise ValueError(
            "training availability contains subjects outside its provenance: "
            f"{sorted(unknown_subjects)}"
        )
    selected = table.loc[
        table["receiver"].map(lambda value: str(value) == receiver),
        [
            "sample_id",
            "subject_id",
            "interaction_id",
            "receptor_availability",
        ],
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected["subject_id"] = selected["subject_id"].astype(str)
    selected["interaction_id"] = selected["interaction_id"].astype(str)
    selected["driver_id"] = selected["interaction_id"].map(dict(driver_by_interaction))
    selected["receptor_availability"] = pd.to_numeric(
        selected["receptor_availability"], errors="coerce"
    )
    selected = selected.dropna(subset=["driver_id", "receptor_availability"])
    if not selected.empty:
        receptor_values = selected["receptor_availability"].to_numpy(dtype=float)
        if np.any(~np.isfinite(receptor_values)) or np.any(
            (receptor_values < 0) | (receptor_values > 1)
        ):
            raise ValueError(
                "training receptor_availability values must be finite in [0, 1]"
            )
        selected["driver_id"] = selected["driver_id"].astype(str)
        per_interaction = (
            selected.groupby(
                ["sample_id", "subject_id", "driver_id", "interaction_id"],
                observed=True,
                sort=True,
            )["receptor_availability"]
            .max()
            .reset_index()
        )
    else:
        per_interaction = pd.DataFrame(
            columns=[
                "sample_id",
                "subject_id",
                "driver_id",
                "interaction_id",
                "receptor_availability",
            ]
        )
    evidence_rows = [
        {
            "driver_id": str(row.driver_id),
            "interaction_id": str(row.interaction_id),
            "receptor_availability": float(
                np.asarray(row.receptor_availability, dtype=np.float64)
            ),
            "sample_id": str(row.sample_id),
            "subject_id": str(row.subject_id),
        }
        for row in per_interaction.itertuples(index=False)
    ]
    evidence_digest = stable_id(
        "training_receptor_evidence",
        {
            "filter_universe_id": availability.filter_universe_id,
            "pooling_method": _POOLING_METHOD,
            "receiver": receiver,
            "rows": evidence_rows,
            "training_subject_ids": list(availability.application_subject_ids),
        },
    )
    gates = dict.fromkeys(prior.driver_ids, 0.0)
    if not per_interaction.empty:
        per_sample = per_interaction.groupby(
            ["sample_id", "subject_id", "driver_id"],
            observed=True,
            sort=True,
        )["receptor_availability"].max()
        pooled = per_sample.groupby("driver_id", observed=True, sort=True).mean()
        for driver, value in pooled.items():
            gates[str(driver)] = float(np.clip(float(value), 0.0, 1.0))
    return tuple(
        (driver, gates[driver]) for driver in prior.driver_ids
    ), evidence_digest


@dataclass(frozen=True, slots=True, init=False)
class ReceiverFamilyTrainingArtifact:
    """Producer-owned partial artifact learned only from one training fold."""

    receiver: str
    fold_id: str
    training_subject_ids: tuple[str, ...]
    filter_universe_id: str
    prior_manifest_digest: str
    driver_by_interaction: tuple[tuple[str, str], ...]
    receptor_gates: tuple[tuple[str, float], ...]
    receptor_evidence_digest: str
    receptor_gate_manifest_id: str
    source_basis: GatedTargetBasis
    family_basis: FamilyFirstBasis
    completed_stages: tuple[str, ...]
    remaining_stages: tuple[str, ...]
    certification_status: str
    training_artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverFamilyTrainingArtifact is producer-owned; "
            "use fit_receiver_family_training_artifact()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        receiver: str,
        fold_id: str,
        training_subject_ids: tuple[str, ...],
        filter_universe_id: str,
        prior_manifest_digest: str,
        driver_by_interaction: tuple[tuple[str, str], ...],
        receptor_gates: tuple[tuple[str, float], ...],
        receptor_evidence_digest: str,
        source_basis: GatedTargetBasis,
        family_basis: FamilyFirstBasis,
    ) -> ReceiverFamilyTrainingArtifact:
        subjects = tuple(sorted(training_subject_ids))
        if not subjects or len(subjects) != len(set(subjects)):
            raise ValueError("training_subject_ids must be non-empty and unique")
        if source_basis.basis_id != family_basis.source_basis_id:
            raise ValueError(
                "family basis does not derive from the supplied source basis"
            )
        if tuple(driver for driver, _ in receptor_gates) != source_basis.driver_ids:
            raise ValueError("receptor gates do not align with source-basis drivers")
        gate_manifest_id = stable_id(
            "receiver_gate_manifest",
            {
                "filter_universe_id": filter_universe_id,
                "fold_id": fold_id,
                "gate_policy": source_basis.gate_policy.value,
                "gate_threshold": source_basis.receptor_gate_threshold,
                "pooling_method": _POOLING_METHOD,
                "prior_manifest_digest": prior_manifest_digest,
                "receiver": receiver,
                "receptor_evidence_digest": receptor_evidence_digest,
                "receptor_gates": [list(item) for item in receptor_gates],
                "training_subject_ids": list(subjects),
            },
        )
        artifact_id = stable_id(
            "receiver_family_training_artifact",
            {
                "completed_stages": list(_COMPLETED_STAGES),
                "driver_by_interaction": [list(item) for item in driver_by_interaction],
                "family_basis_id": family_basis.family_basis_id,
                "filter_universe_id": filter_universe_id,
                "fold_id": fold_id,
                "prior_manifest_digest": prior_manifest_digest,
                "receiver": receiver,
                "receptor_gate_manifest_id": gate_manifest_id,
                "source_basis_id": source_basis.basis_id,
                "training_subject_ids": list(subjects),
            },
        )
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "receiver": receiver,
            "fold_id": fold_id,
            "training_subject_ids": subjects,
            "filter_universe_id": filter_universe_id,
            "prior_manifest_digest": prior_manifest_digest,
            "driver_by_interaction": driver_by_interaction,
            "receptor_gates": receptor_gates,
            "receptor_evidence_digest": receptor_evidence_digest,
            "receptor_gate_manifest_id": gate_manifest_id,
            "source_basis": source_basis,
            "family_basis": family_basis,
            "completed_stages": _COMPLETED_STAGES,
            "remaining_stages": _REMAINING_STAGES,
            "certification_status": _TRAINING_STATUS,
            "training_artifact_id": artifact_id,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    @property
    def eligible_family_ids(self) -> tuple[str, ...]:
        """Return family IDs whose training-only receptor gate is eligible."""

        return tuple(
            family_id
            for family_id, eligible in zip(
                self.family_basis.family_ids,
                self.family_basis.family_eligible,
                strict=True,
            )
            if bool(eligible)
        )

    @property
    def is_oof_certified(self) -> bool:
        """Return false because response fitting and common scoring remain."""

        return False

    def _require_producer_owned(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("receiver-family artifact was not produced by this module")


def fit_receiver_family_training_artifact(
    availability: BatchAvailability,
    prior: TargetPrior,
    *,
    receiver: str,
    fold_id: str,
    feature_ids: Sequence[str],
    driver_by_interaction: Mapping[str, str],
    receptor_gate_threshold: float = 0.1,
    cosine_threshold: float = 0.95,
) -> ReceiverFamilyTrainingArtifact:
    """Freeze one receiver by delegating to the fold-level batch producer."""

    return fit_receiver_family_training_artifacts(
        availability,
        prior,
        receivers=(receiver,),
        fold_id=fold_id,
        feature_ids=feature_ids,
        driver_by_interaction=driver_by_interaction,
        receptor_gate_threshold=receptor_gate_threshold,
        cosine_threshold=cosine_threshold,
    )[0]


def fit_receiver_family_training_artifacts(
    availability: BatchAvailability,
    prior: TargetPrior,
    *,
    receivers: Sequence[str],
    fold_id: str,
    feature_ids: Sequence[str],
    driver_by_interaction: Mapping[str, str],
    receptor_gate_threshold: float = 0.1,
    cosine_threshold: float = 0.95,
) -> tuple[ReceiverFamilyTrainingArtifact, ...]:
    """Freeze receiver gates while sharing one static family partition per fold.

    Strict driver families depend only on the normalized target-prior profiles,
    not on receiver-specific receptor gates.  This producer therefore builds a
    gated source basis for every receiver, verifies that their ungated profiles
    are byte-for-byte identical, and clusters the shared profile universe once.
    Each returned artifact still owns its receiver-specific gates, evidence,
    source basis, family basis, and identity.
    """

    if not isinstance(availability, BatchAvailability):
        raise TypeError("availability must be a BatchAvailability")
    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    if (
        availability.filter_application
        is not InteractionFilterApplication.TRAINING_SELECTION_V1
    ):
        raise ValueError(
            "receiver-family fitting requires training-selection availability"
        )
    if isinstance(receivers, (str, bytes)):
        raise TypeError("receivers must be a sequence of receiver names")
    normalized_receivers = tuple(
        sorted(
            _nonempty_name(receiver, field_name="receiver")
            for receiver in receivers
        )
    )
    if not normalized_receivers:
        raise ValueError("receivers must contain at least one receiver")
    if len(normalized_receivers) != len(set(normalized_receivers)):
        raise ValueError("receivers must be unique")
    normalized_fold = _nonempty_name(fold_id, field_name="fold_id")
    threshold = float(receptor_gate_threshold)
    cosine = float(cosine_threshold)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("receptor_gate_threshold must be finite in (0, 1]")
    if not math.isfinite(cosine) or not 0 <= cosine <= 1:
        raise ValueError("cosine_threshold must be finite in [0, 1]")
    mapping = _driver_mapping(driver_by_interaction, prior)
    receiver_inputs: list[
        tuple[str, tuple[tuple[str, float], ...], str, GatedTargetBasis]
    ] = []
    for normalized_receiver in normalized_receivers:
        receptor_gates, evidence_digest = _pooled_receptor_gates(
            availability,
            receiver=normalized_receiver,
            prior=prior,
            driver_by_interaction=mapping,
        )
        source_basis = build_gated_target_basis(
            prior,
            feature_ids,
            dict(receptor_gates),
            gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
            receptor_gate_threshold=threshold,
        )
        receiver_inputs.append(
            (
                normalized_receiver,
                receptor_gates,
                evidence_digest,
                source_basis,
            )
        )

    shared_source_basis = receiver_inputs[0][3]
    shared_profiles = shared_source_basis.normalized_profiles
    for receiver_name, _, _, source_basis in receiver_inputs[1:]:
        profiles = source_basis.normalized_profiles
        if (
            source_basis.feature_ids != shared_source_basis.feature_ids
            or source_basis.driver_ids != shared_source_basis.driver_ids
            or profiles.shape != shared_profiles.shape
            or not np.array_equal(profiles.indptr, shared_profiles.indptr)
            or not np.array_equal(profiles.indices, shared_profiles.indices)
            or not np.array_equal(profiles.data, shared_profiles.data)
        ):
            raise RuntimeError(
                "receiver source bases disagree on normalized target profiles: "
                f"{receiver_name!r}"
            )
    families = cluster_driver_families(
        shared_source_basis,
        cosine_threshold=cosine,
    )
    artifacts: list[ReceiverFamilyTrainingArtifact] = []
    for receiver_name, receptor_gates, evidence_digest, source_basis in receiver_inputs:
        family_basis = build_family_first_basis(
            source_basis,
            families,
            strict_cosine_threshold=cosine,
        )
        artifacts.append(
            ReceiverFamilyTrainingArtifact._from_training(
                receiver=receiver_name,
                fold_id=normalized_fold,
                training_subject_ids=availability.application_subject_ids,
                filter_universe_id=availability.filter_universe_id,
                prior_manifest_digest=prior.manifest_digest,
                driver_by_interaction=mapping,
                receptor_gates=receptor_gates,
                receptor_evidence_digest=evidence_digest,
                source_basis=source_basis,
                family_basis=family_basis,
            )
        )
    return tuple(artifacts)


__all__ = [
    "ReceiverFamilyTrainingArtifact",
    "fit_receiver_family_training_artifact",
    "fit_receiver_family_training_artifacts",
]
