"""Training-only receptor gates and strict receiver-family bases."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from crychic.availability import (
    BatchAvailability,
    InteractionFilterApplication,
)
from crychic.core import ContractError, stable_id
from crychic.resources import TargetPrior

from .basis import build_gated_target_basis
from .contracts import (
    BasisBuildReport,
    DriverFamilyDefinition,
    FamilyFirstBasis,
    GatedTargetBasis,
    ReceptorGatePolicy,
)
from .families import cluster_driver_families
from .family_first import build_family_first_basis

_PRODUCER_MARKER = "crychic.receiver_family_training.v2"
_TRAINING_STATUS = "training_only_partial_receiver_family_v1"
_POOLING_METHOD = (
    "condition_blind_sample_interaction_max_then_sample_driver_max_then_"
    "equal_context_subject_mean_then_equal_subject_mean_v2"
)
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
    "context_id",
    "receiver",
    "interaction_id",
    "receptor_availability",
}


def _immutable_array(values: np.ndarray, *, dtype: str) -> np.ndarray:
    canonical = np.asarray(values, dtype=dtype, order="C")
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype=dtype).reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _immutable_csc(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    canonical = sparse.csc_matrix(matrix, dtype="<f8", copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.data = _immutable_array(canonical.data, dtype="<f8")
    canonical.indices = _immutable_array(canonical.indices, dtype="<i8")
    canonical.indptr = _immutable_array(canonical.indptr, dtype="<i8")
    return canonical


def _is_immutable_byte_backed(values: np.ndarray) -> bool:
    if values.flags.writeable or not values.flags.c_contiguous:
        return False
    base: object = values
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _array_digest(values: np.ndarray, *, dtype: str) -> str:
    canonical = np.asarray(values, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _csc_digest(matrix: sparse.spmatrix) -> str:
    canonical = sparse.csc_matrix(matrix, dtype="<f8", copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.data, dtype="<f8").tobytes())
    digest.update(np.asarray(canonical.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.indptr, dtype="<i8").tobytes())
    return digest.hexdigest()


def _freeze_source_basis(basis: GatedTargetBasis) -> GatedTargetBasis:
    report = basis.report
    frozen = GatedTargetBasis(
        basis_id=basis.basis_id,
        feature_ids=tuple(basis.feature_ids),
        driver_ids=tuple(basis.driver_ids),
        normalized_profiles=basis.normalized_profiles,
        matrix=basis.matrix,
        receptor_gates=basis.receptor_gates,
        receptor_eligible=basis.receptor_eligible,
        gate_policy=basis.gate_policy,
        receptor_gate_threshold=basis.receptor_gate_threshold,
        pre_normalization_norms=basis.pre_normalization_norms,
        prior_resource_id=basis.prior_resource_id,
        prior_version=basis.prior_version,
        report=BasisBuildReport(
            prior_targets=report.prior_targets,
            response_features=report.response_features,
            matched_targets=report.matched_targets,
            unmatched_prior_targets=tuple(report.unmatched_prior_targets),
            zero_norm_drivers=tuple(report.zero_norm_drivers),
            zero_gate_drivers=tuple(report.zero_gate_drivers),
        ),
    )
    object.__setattr__(
        frozen, "normalized_profiles", _immutable_csc(frozen.normalized_profiles)
    )
    object.__setattr__(frozen, "matrix", _immutable_csc(frozen.matrix))
    object.__setattr__(
        frozen,
        "receptor_gates",
        _immutable_array(frozen.receptor_gates, dtype="<f8"),
    )
    object.__setattr__(
        frozen,
        "receptor_eligible",
        _immutable_array(frozen.receptor_eligible, dtype="|b1"),
    )
    object.__setattr__(
        frozen,
        "pre_normalization_norms",
        _immutable_array(frozen.pre_normalization_norms, dtype="<f8"),
    )
    return frozen


def _freeze_family_basis(basis: FamilyFirstBasis) -> FamilyFirstBasis:
    definitions = tuple(
        DriverFamilyDefinition(
            family_id=family.family_id,
            driver_ids=tuple(family.driver_ids),
            mean_pairwise_cosine=family.mean_pairwise_cosine,
            assignment_uncertainty=family.assignment_uncertainty,
        )
        for family in basis.family_definitions
    )
    frozen = FamilyFirstBasis(
        family_basis_id=basis.family_basis_id,
        source_basis_id=basis.source_basis_id,
        feature_ids=tuple(basis.feature_ids),
        family_definitions=definitions,
        matrix=basis.matrix,
        medoid_driver_ids=tuple(basis.medoid_driver_ids),
        family_eligible=basis.family_eligible,
        strict_cosine_threshold=basis.strict_cosine_threshold,
        method=basis.method,
        experimental=basis.experimental,
    )
    object.__setattr__(frozen, "matrix", _immutable_csc(frozen.matrix))
    object.__setattr__(
        frozen,
        "family_eligible",
        _immutable_array(frozen.family_eligible, dtype="|b1"),
    )
    return frozen


def _source_basis_payload(basis: GatedTargetBasis) -> dict[str, object]:
    report = basis.report
    return {
        "basis_id": basis.basis_id,
        "driver_ids": list(basis.driver_ids),
        "feature_ids": list(basis.feature_ids),
        "gate_policy": basis.gate_policy.value,
        "matrix_digest": _csc_digest(basis.matrix),
        "normalized_profiles_digest": _csc_digest(basis.normalized_profiles),
        "pre_normalization_norms_digest": _array_digest(
            basis.pre_normalization_norms, dtype="<f8"
        ),
        "prior_resource_id": basis.prior_resource_id,
        "prior_version": basis.prior_version,
        "receptor_eligible_digest": _array_digest(basis.receptor_eligible, dtype="|b1"),
        "receptor_gate_threshold": basis.receptor_gate_threshold,
        "receptor_gates_digest": _array_digest(basis.receptor_gates, dtype="<f8"),
        "report": {
            "matched_targets": report.matched_targets,
            "prior_targets": report.prior_targets,
            "response_features": report.response_features,
            "unmatched_prior_targets": list(report.unmatched_prior_targets),
            "zero_gate_drivers": list(report.zero_gate_drivers),
            "zero_norm_drivers": list(report.zero_norm_drivers),
        },
    }


def _family_basis_payload(basis: FamilyFirstBasis) -> dict[str, object]:
    return {
        "experimental": basis.experimental,
        "family_basis_id": basis.family_basis_id,
        "family_definitions": [
            {
                "assignment_uncertainty": family.assignment_uncertainty,
                "driver_ids": list(family.driver_ids),
                "family_id": family.family_id,
                "mean_pairwise_cosine": family.mean_pairwise_cosine,
            }
            for family in basis.family_definitions
        ],
        "family_eligible_digest": _array_digest(basis.family_eligible, dtype="|b1"),
        "feature_ids": list(basis.feature_ids),
        "matrix_digest": _csc_digest(basis.matrix),
        "medoid_driver_ids": list(basis.medoid_driver_ids),
        "method": basis.method.value,
        "source_basis_id": basis.source_basis_id,
        "strict_cosine_threshold": basis.strict_cosine_threshold,
    }


def _basis_digest(kind: str, payload: dict[str, object]) -> str:
    return stable_id(kind, payload, schema_version="2", digest_length=64)


def _gate_manifest_id(
    *,
    filter_universe_id: str,
    fold_id: str,
    prior_manifest_digest: str,
    receiver: str,
    receptor_evidence_digest: str,
    receptor_gates: tuple[tuple[str, float], ...],
    source_basis: GatedTargetBasis,
    training_subject_ids: tuple[str, ...],
) -> str:
    return stable_id(
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
            "training_subject_ids": list(training_subject_ids),
        },
    )


def _training_artifact_id(
    *,
    receiver: str,
    fold_id: str,
    training_subject_ids: tuple[str, ...],
    filter_universe_id: str,
    prior_manifest_digest: str,
    driver_by_interaction: tuple[tuple[str, str], ...],
    receptor_gates: tuple[tuple[str, float], ...],
    receptor_evidence_digest: str,
    receptor_gate_manifest_id: str,
    source_basis_digest: str,
    family_basis_digest: str,
) -> str:
    return stable_id(
        "receiver_family_training_artifact",
        {
            "certification_status": _TRAINING_STATUS,
            "completed_stages": list(_COMPLETED_STAGES),
            "driver_by_interaction": [list(item) for item in driver_by_interaction],
            "family_basis_digest": family_basis_digest,
            "filter_universe_id": filter_universe_id,
            "fold_id": fold_id,
            "prior_manifest_digest": prior_manifest_digest,
            "receiver": receiver,
            "receptor_evidence_digest": receptor_evidence_digest,
            "receptor_gate_manifest_id": receptor_gate_manifest_id,
            "receptor_gates": [list(item) for item in receptor_gates],
            "remaining_stages": list(_REMAINING_STAGES),
            "source_basis_digest": source_basis_digest,
            "training_subject_ids": list(training_subject_ids),
        },
        schema_version="2",
    )


def _validate_basis_relationship(
    *,
    source_basis: GatedTargetBasis,
    family_basis: FamilyFirstBasis,
    prior_manifest_digest: str,
) -> None:
    if source_basis.basis_id != family_basis.source_basis_id:
        raise ValueError("family basis does not derive from the supplied source basis")
    if source_basis.feature_ids != family_basis.feature_ids:
        raise ValueError("family basis feature order does not match source basis")
    if family_basis.driver_ids != source_basis.driver_ids:
        raise ValueError("family definitions do not partition source-basis drivers")
    expected_source_id = stable_id(
        "target_basis",
        {
            "driver_ids": source_basis.driver_ids,
            "feature_ids": source_basis.feature_ids,
            "gates": source_basis.receptor_gates.tolist(),
            "gate_policy": source_basis.gate_policy.value,
            "gate_policy_version": source_basis.gate_policy.version,
            "receptor_gate_threshold": source_basis.receptor_gate_threshold,
            "receptor_eligible": source_basis.receptor_eligible.tolist(),
            "prior_manifest": prior_manifest_digest,
            "prior_resource_id": source_basis.prior_resource_id,
            "prior_version": source_basis.prior_version,
        },
    )
    if source_basis.basis_id != expected_source_id:
        raise ValueError("source basis ID does not match its complete manifest")
    driver_index = {
        driver: index for index, driver in enumerate(source_basis.driver_ids)
    }
    expected_eligible = np.asarray(
        [
            (
                float(
                    sparse.linalg.norm(
                        source_basis.normalized_profiles[:, driver_index[medoid]]
                    )
                )
                > 0
                and any(
                    bool(source_basis.receptor_eligible[driver_index[driver]])
                    for driver in family.driver_ids
                )
            )
            for family, medoid in zip(
                family_basis.family_definitions,
                family_basis.medoid_driver_ids,
                strict=True,
            )
        ],
        dtype=bool,
    )
    if not np.array_equal(family_basis.family_eligible, expected_eligible):
        raise ValueError("family eligibility does not derive from source gates")
    expected_columns = [
        (
            source_basis.normalized_profiles[:, driver_index[medoid]]
            if eligible
            else sparse.csc_matrix((len(source_basis.feature_ids), 1), dtype=float)
        )
        for medoid, eligible in zip(
            family_basis.medoid_driver_ids, expected_eligible, strict=True
        )
    ]
    expected_matrix = sparse.hstack(expected_columns, format="csc")
    difference = (family_basis.matrix - expected_matrix).tocsc()
    if difference.nnz and not np.allclose(difference.data, 0.0, rtol=1e-12, atol=1e-14):
        raise ValueError("family matrix does not match source medoid columns")
    expected_family_id = stable_id(
        "family_first_basis",
        {
            "source_basis_id": source_basis.basis_id,
            "method": family_basis.method.value,
            "method_version": family_basis.method.version,
            "strict_cosine_threshold": family_basis.strict_cosine_threshold,
            "families": [
                {
                    "family_id": family.family_id,
                    "driver_ids": family.driver_ids,
                    "medoid_driver_id": medoid,
                    "eligible": bool(eligible),
                }
                for family, medoid, eligible in zip(
                    family_basis.family_definitions,
                    family_basis.medoid_driver_ids,
                    family_basis.family_eligible,
                    strict=True,
                )
            ],
        },
    )
    if family_basis.family_basis_id != expected_family_id:
        raise ValueError("family basis ID does not match its manifest")


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
            "context_id",
            "interaction_id",
            "receptor_availability",
        ],
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected["subject_id"] = selected["subject_id"].astype(str)
    selected["context_id"] = selected["context_id"].astype(str)
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
                [
                    "sample_id",
                    "subject_id",
                    "context_id",
                    "driver_id",
                    "interaction_id",
                ],
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
                "context_id",
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
            "context_id": str(row.context_id),
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
            ["sample_id", "subject_id", "context_id", "driver_id"],
            observed=True,
            sort=True,
        )["receptor_availability"].max()
        per_subject_context = per_sample.groupby(
            ["subject_id", "context_id", "driver_id"],
            observed=True,
            sort=True,
        ).mean()
        per_subject = per_subject_context.groupby(
            ["subject_id", "driver_id"],
            observed=True,
            sort=True,
        ).mean()
        pooled = per_subject.groupby("driver_id", observed=True, sort=True).mean()
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
    source_basis_digest: str
    family_basis: FamilyFirstBasis
    family_basis_digest: str
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
        frozen_source = _freeze_source_basis(source_basis)
        frozen_family = _freeze_family_basis(family_basis)
        _validate_basis_relationship(
            source_basis=frozen_source,
            family_basis=frozen_family,
            prior_manifest_digest=prior_manifest_digest,
        )
        if tuple(driver for driver, _ in receptor_gates) != frozen_source.driver_ids:
            raise ValueError("receptor gates do not align with source-basis drivers")
        if not np.array_equal(
            np.asarray([value for _, value in receptor_gates], dtype=float),
            frozen_source.receptor_gates,
        ):
            raise ValueError("receptor gates do not match source-basis gate values")
        gate_manifest_id = _gate_manifest_id(
            filter_universe_id=filter_universe_id,
            fold_id=fold_id,
            prior_manifest_digest=prior_manifest_digest,
            receiver=receiver,
            receptor_evidence_digest=receptor_evidence_digest,
            receptor_gates=receptor_gates,
            source_basis=frozen_source,
            training_subject_ids=subjects,
        )
        source_digest = _basis_digest(
            "receiver_source_basis", _source_basis_payload(frozen_source)
        )
        family_digest = _basis_digest(
            "receiver_family_basis", _family_basis_payload(frozen_family)
        )
        artifact_id = _training_artifact_id(
            receiver=receiver,
            fold_id=fold_id,
            training_subject_ids=subjects,
            filter_universe_id=filter_universe_id,
            prior_manifest_digest=prior_manifest_digest,
            driver_by_interaction=driver_by_interaction,
            receptor_gates=receptor_gates,
            receptor_evidence_digest=receptor_evidence_digest,
            receptor_gate_manifest_id=gate_manifest_id,
            source_basis_digest=source_digest,
            family_basis_digest=family_digest,
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
            "source_basis": frozen_source,
            "source_basis_digest": source_digest,
            "family_basis": frozen_family,
            "family_basis_digest": family_digest,
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
        try:
            subjects = tuple(self.training_subject_ids)
            if (
                not subjects
                or subjects != tuple(sorted(subjects))
                or len(subjects) != len(set(subjects))
            ):
                raise ValueError("invalid training subjects")
            if self.completed_stages != _COMPLETED_STAGES:
                raise ValueError("invalid completed stages")
            if self.remaining_stages != _REMAINING_STAGES:
                raise ValueError("invalid remaining stages")
            if self.certification_status != _TRAINING_STATUS:
                raise ValueError("invalid certification status")
            if tuple(sorted(self.driver_by_interaction)) != self.driver_by_interaction:
                raise ValueError("invalid driver mapping order")
            if tuple(driver for driver, _ in self.receptor_gates) != (
                self.source_basis.driver_ids
            ):
                raise ValueError("invalid receptor-gate order")
            if not np.array_equal(
                np.asarray([value for _, value in self.receptor_gates], dtype=float),
                self.source_basis.receptor_gates,
            ):
                raise ValueError("invalid receptor-gate values")
            arrays = (
                self.source_basis.normalized_profiles.data,
                self.source_basis.normalized_profiles.indices,
                self.source_basis.normalized_profiles.indptr,
                self.source_basis.matrix.data,
                self.source_basis.matrix.indices,
                self.source_basis.matrix.indptr,
                self.source_basis.receptor_gates,
                self.source_basis.receptor_eligible,
                self.source_basis.pre_normalization_norms,
                self.family_basis.matrix.data,
                self.family_basis.matrix.indices,
                self.family_basis.matrix.indptr,
                self.family_basis.family_eligible,
            )
            if not all(_is_immutable_byte_backed(array) for array in arrays):
                raise ValueError("basis buffers are not immutable")
            _validate_basis_relationship(
                source_basis=self.source_basis,
                family_basis=self.family_basis,
                prior_manifest_digest=self.prior_manifest_digest,
            )
            gate_manifest_id = _gate_manifest_id(
                filter_universe_id=self.filter_universe_id,
                fold_id=self.fold_id,
                prior_manifest_digest=self.prior_manifest_digest,
                receiver=self.receiver,
                receptor_evidence_digest=self.receptor_evidence_digest,
                receptor_gates=self.receptor_gates,
                source_basis=self.source_basis,
                training_subject_ids=subjects,
            )
            source_digest = _basis_digest(
                "receiver_source_basis", _source_basis_payload(self.source_basis)
            )
            family_digest = _basis_digest(
                "receiver_family_basis", _family_basis_payload(self.family_basis)
            )
            expected_id = _training_artifact_id(
                receiver=self.receiver,
                fold_id=self.fold_id,
                training_subject_ids=subjects,
                filter_universe_id=self.filter_universe_id,
                prior_manifest_digest=self.prior_manifest_digest,
                driver_by_interaction=self.driver_by_interaction,
                receptor_gates=self.receptor_gates,
                receptor_evidence_digest=self.receptor_evidence_digest,
                receptor_gate_manifest_id=gate_manifest_id,
                source_basis_digest=source_digest,
                family_basis_digest=family_digest,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-family artifact failed integrity validation",
                code="receiver_family_integrity_violation",
                field="training_artifact_id",
                remediation="Refit the receiver-family artifact from training data",
            ) from error
        if (
            gate_manifest_id != self.receptor_gate_manifest_id
            or source_digest != self.source_basis_digest
            or family_digest != self.family_basis_digest
            or expected_id != self.training_artifact_id
        ):
            raise ContractError(
                "Receiver-family artifact provenance failed integrity validation",
                code="receiver_family_integrity_violation",
                field="training_artifact_id",
                remediation="Refit the receiver-family artifact from training data",
            )


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
            _nonempty_name(receiver, field_name="receiver") for receiver in receivers
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
