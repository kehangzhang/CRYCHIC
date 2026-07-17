"""Training-only receptor gates and strict receiver-family bases."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import sparse

from crychic.availability import (
    BatchAvailability,
    InteractionFilterApplication,
)
from crychic.core import CommunicationMode, ContractError, stable_id
from crychic.resources import (
    FrozenMolecularLREquivalenceUniverse,
    MolecularLRMappingStatus,
    ResourceBundle,
    TargetPrior,
    freeze_molecular_lr_equivalence_universe,
)

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
_RECEIVER_FAMILY_UNIVERSE_PRODUCER = (
    "crychic.frozen_receiver_family_opportunity_universe.v1"
)
_RECEIVER_FAMILY_UNIVERSE_POLICY = (
    "target_prior_root_feature_axis_context_label_blind_v1"
)
_RECEIVER_FAMILY_UNIVERSE_REUSE_PRODUCER = (
    "crychic.receiver_family_opportunity_universe_reuse_binding.v1"
)
_RECEIVER_FAMILY_LR_UNIVERSE_PRODUCER = (
    "crychic.frozen_receiver_family_lr_hypothesis_universe.v2"
)
_RECEIVER_FAMILY_LR_UNIVERSE_POLICY = (
    "external_resource_molecular_equivalence_target_prior_mapping_v2"
)
_RECEIVER_FAMILY_LR_MAPPING_POLICY = (
    "resource_molecular_equivalence_prior_unique_driver_match_v2"
)
_RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION = "2.0.0"
_FROZEN_AXIS_REFIT_PRODUCER_MARKER = "crychic.receiver_family_frozen_axis_refit.v1"
_FROZEN_AXIS_REFIT_METHOD = "outer_axis_inner_train_receptor_gate_refit_v1"
_FROZEN_AXIS_INSUFFICIENT_TRAINING_SUBJECTS = (
    "receiver_family_frozen_axis_insufficient_training_subjects"
)
_FROZEN_AXIS_INSUFFICIENT_RECEIVER_SUBJECTS = (
    "receiver_family_frozen_axis_insufficient_receiver_subjects"
)
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

ReceiverFamilyAxisRefitStatus = Literal["observed", "not_estimable"]


def _canonical_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    names = tuple(
        sorted(_nonempty_name(value, field_name=field_name) for value in values)
    )
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return names


def _ordered_unique_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    names = tuple(_nonempty_name(value, field_name=field_name) for value in values)
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return names


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
    return str(stable_id(kind, payload, schema_version="2", digest_length=64))


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
    return str(
        stable_id(
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
    frozen_family_axis_parent_id: str | None = None,
) -> str:
    payload: dict[str, object] = {
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
    }
    if frozen_family_axis_parent_id is not None:
        payload["frozen_family_axis_parent_id"] = frozen_family_axis_parent_id
    return str(
        stable_id(
            "receiver_family_training_artifact",
            payload,
            schema_version="2",
        )
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


def _subject_ids(
    values: Sequence[str], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(
        sorted(_nonempty_name(value, field_name=field_name) for value in values)
    )
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique subjects")
    return result


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


@dataclass(frozen=True, slots=True)
class _PooledReceptorGateFit:
    receptor_gates: tuple[tuple[str, float], ...]
    evidence_digest: str
    evidence_subject_ids: tuple[str, ...]
    evidence_row_count: int


def _pooled_receptor_gates(
    availability: BatchAvailability,
    *,
    receiver: str,
    driver_ids: tuple[str, ...],
    driver_by_interaction: tuple[tuple[str, str], ...],
) -> _PooledReceptorGateFit:
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
    evidence_digest = str(
        stable_id(
            "training_receptor_evidence",
            {
                "filter_universe_id": availability.filter_universe_id,
                "pooling_method": _POOLING_METHOD,
                "receiver": receiver,
                "rows": evidence_rows,
                "training_subject_ids": list(availability.application_subject_ids),
            },
        )
    )
    gates = dict.fromkeys(driver_ids, 0.0)
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
    return _PooledReceptorGateFit(
        receptor_gates=tuple((driver, gates[driver]) for driver in driver_ids),
        evidence_digest=evidence_digest,
        evidence_subject_ids=tuple(
            sorted(set(per_interaction["subject_id"].astype(str)))
        ),
        evidence_row_count=len(per_interaction),
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenReceiverFamilyOpportunityUniverse:
    """Run-root receiver x strict-family opportunities frozen before fitting.

    Strict families are derived from the external target prior projected onto
    the context-label-blind root feature axis.  Receiver expression, receptor
    gates, fold membership, and response values are deliberately absent from
    this contract.  The universe therefore owns opportunities for a receiver
    even when that receiver is absent from an outer-training fold.
    """

    receiver_ids: tuple[str, ...]
    receiver_universe_id: str
    receiver_axis_id: str
    feature_ids: tuple[str, ...]
    feature_axis_id: str
    driver_ids: tuple[str, ...]
    family_definitions: tuple[DriverFamilyDefinition, ...]
    family_ids: tuple[str, ...]
    family_axis_id: str
    opportunity_ids: tuple[tuple[str, str, str], ...]
    opportunity_axis_id: str
    prior_resource_id: str
    prior_version: str
    prior_manifest_digest: str
    prior_content_id: str
    root_input_identity_id: str
    root_input_digest: str
    cosine_threshold: float
    source_policy: str
    universe_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenReceiverFamilyOpportunityUniverse is producer-owned; use "
            "freeze_receiver_family_opportunity_universe()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "cosine_threshold": self.cosine_threshold,
            "driver_ids": list(self.driver_ids),
            "family_axis_id": self.family_axis_id,
            "family_definitions": [
                {
                    "assignment_uncertainty": family.assignment_uncertainty,
                    "driver_ids": list(family.driver_ids),
                    "family_id": family.family_id,
                    "mean_pairwise_cosine": family.mean_pairwise_cosine,
                }
                for family in self.family_definitions
            ],
            "feature_axis_id": self.feature_axis_id,
            "opportunity_axis_id": self.opportunity_axis_id,
            "prior_manifest_digest": self.prior_manifest_digest,
            "prior_content_id": self.prior_content_id,
            "prior_resource_id": self.prior_resource_id,
            "prior_version": self.prior_version,
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_universe_id": self.receiver_universe_id,
            "root_input_digest": self.root_input_digest,
            "root_input_identity_id": self.root_input_identity_id,
            "source_policy": self.source_policy,
        }

    def _require_intact(self) -> None:
        try:
            receivers = _canonical_names(self.receiver_ids, field_name="receiver_ids")
            features = _ordered_unique_names(self.feature_ids, field_name="feature_ids")
            drivers = _canonical_names(self.driver_ids, field_name="driver_ids")
            definitions = tuple(self.family_definitions)
            if (
                not definitions
                or definitions
                != tuple(sorted(definitions, key=lambda family: family.family_id))
                or len({family.family_id for family in definitions}) != len(definitions)
            ):
                raise ValueError("family definitions are not canonical")
            members = tuple(
                sorted(driver for family in definitions for driver in family.driver_ids)
            )
            if members != drivers or len(members) != len(set(members)):
                raise ValueError("family definitions do not partition drivers")
            for family in definitions:
                expected_family_id = stable_id(
                    "driver_family",
                    {
                        "driver_ids": family.driver_ids,
                        "prior_resource_id": self.prior_resource_id,
                        "prior_version": self.prior_version,
                    },
                )
                if family.family_id != expected_family_id:
                    raise ValueError("family identity differs from its prior members")
            families = tuple(family.family_id for family in definitions)
            threshold = float(self.cosine_threshold)
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("cosine_threshold must lie in [0, 1]")
            expected_receiver_axis_id = stable_id(
                "receiver_axis",
                {"receiver_ids": list(receivers)},
                schema_version="1",
            )
            expected_feature_axis_id = stable_id(
                "root_feature_axis",
                {"feature_ids": list(features)},
                schema_version="1",
            )
            expected_family_axis_id = stable_id(
                "strict_driver_family_axis",
                {
                    "cosine_threshold": threshold,
                    "family_definitions": [
                        {
                            "driver_ids": list(family.driver_ids),
                            "family_id": family.family_id,
                        }
                        for family in definitions
                    ],
                    "feature_axis_id": expected_feature_axis_id,
                    "prior_manifest_digest": self.prior_manifest_digest,
                },
                schema_version="1",
            )
            expected_opportunity_axis_id = stable_id(
                "receiver_family_opportunity_axis",
                {
                    "family_axis_id": expected_family_axis_id,
                    "receiver_axis_id": expected_receiver_axis_id,
                },
                schema_version="1",
            )
            expected_opportunities = tuple(
                (
                    receiver,
                    family_id,
                    stable_id(
                        "receiver_family_opportunity",
                        {
                            "family_id": family_id,
                            "opportunity_axis_id": expected_opportunity_axis_id,
                            "receiver": receiver,
                        },
                        schema_version="1",
                    ),
                )
                for receiver in receivers
                for family_id in families
            )
            expected_universe_id = stable_id(
                "frozen_receiver_family_opportunity_universe",
                self._identity_payload(),
                schema_version="1",
            )
            valid = (
                self._producer_marker == _RECEIVER_FAMILY_UNIVERSE_PRODUCER
                and self.source_policy == _RECEIVER_FAMILY_UNIVERSE_POLICY
                and receivers == self.receiver_ids
                and features == self.feature_ids
                and drivers == self.driver_ids
                and families == self.family_ids
                and self.receiver_axis_id == expected_receiver_axis_id
                and self.feature_axis_id == expected_feature_axis_id
                and self.family_axis_id == expected_family_axis_id
                and self.opportunity_axis_id == expected_opportunity_axis_id
                and self.opportunity_ids == expected_opportunities
                and self.universe_id == expected_universe_id
                and all(
                    isinstance(value, str) and bool(value)
                    for value in (
                        self.receiver_universe_id,
                        self.prior_resource_id,
                        self.prior_version,
                        self.prior_manifest_digest,
                        self.prior_content_id,
                        self.root_input_identity_id,
                        self.root_input_digest,
                    )
                )
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen receiver-family opportunity universe failed "
                "integrity validation",
                code="frozen_receiver_family_universe_integrity_violation",
                field="universe_id",
                remediation=(
                    "Refreeze it from the intact target prior, root feature axis, "
                    "and frozen receiver universe"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Frozen receiver-family opportunity universe failed "
                "integrity validation",
                code="frozen_receiver_family_universe_integrity_violation",
                field="universe_id",
                remediation=(
                    "Refreeze it from the intact target prior, root feature axis, "
                    "and frozen receiver universe"
                ),
            )

    def opportunity_id_for(self, receiver: str, family_id: str) -> str:
        """Return one exact frozen receiver-family opportunity identity."""

        self._require_intact()
        key = (
            _nonempty_name(receiver, field_name="receiver"),
            _nonempty_name(family_id, field_name="family_id"),
        )
        by_key = {
            (receiver_name, family_name): opportunity_id
            for receiver_name, family_name, opportunity_id in self.opportunity_ids
        }
        try:
            return by_key[key]
        except KeyError as error:
            raise ContractError(
                "Receiver-family pair is absent from the frozen opportunity universe",
                code="receiver_family_opportunity_not_in_universe",
                field="receiver,family_id",
                remediation="Use one exact receiver-family pair from this universe",
            ) from error

    def to_dict(self) -> dict[str, object]:
        """Return the complete run-root family opportunity manifest."""

        self._require_intact()
        return {
            "universe_id": self.universe_id,
            **self._identity_payload(),
            "receiver_ids": list(self.receiver_ids),
            "feature_ids": list(self.feature_ids),
            "driver_ids": list(self.driver_ids),
            "family_ids": list(self.family_ids),
            "opportunity_count": len(self.opportunity_ids),
            "opportunities": [
                {
                    "receiver": receiver,
                    "family_id": family_id,
                    "opportunity_id": opportunity_id,
                }
                for receiver, family_id, opportunity_id in self.opportunity_ids
            ],
        }


def freeze_receiver_family_opportunity_universe(
    prior: TargetPrior,
    *,
    feature_ids: Sequence[str],
    receiver_ids: Sequence[str],
    receiver_universe_id: str,
    receiver_axis_id: str,
    prior_content_id: str,
    root_input_identity_id: str,
    root_input_digest: str,
    cosine_threshold: float,
) -> FrozenReceiverFamilyOpportunityUniverse:
    """Freeze the prior-owned strict family axis before receiver/fold fitting."""

    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    receivers = _canonical_names(receiver_ids, field_name="receiver_ids")
    features = _ordered_unique_names(feature_ids, field_name="feature_ids")
    threshold = float(cosine_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("cosine_threshold must lie in [0, 1]")
    expected_receiver_axis_id = stable_id(
        "receiver_axis",
        {"receiver_ids": list(receivers)},
        schema_version="1",
    )
    if receiver_axis_id != expected_receiver_axis_id:
        raise ValueError("receiver_axis_id does not match receiver_ids")
    source_basis = build_gated_target_basis(
        prior,
        features,
        dict.fromkeys(prior.driver_ids, 1.0),
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
        receptor_gate_threshold=0.1,
    )
    definitions = cluster_driver_families(
        source_basis,
        cosine_threshold=threshold,
    )
    if not definitions:
        raise ValueError("target prior does not define any strict driver family")
    feature_axis_id = stable_id(
        "root_feature_axis",
        {"feature_ids": list(features)},
        schema_version="1",
    )
    family_axis_id = stable_id(
        "strict_driver_family_axis",
        {
            "cosine_threshold": threshold,
            "family_definitions": [
                {
                    "driver_ids": list(family.driver_ids),
                    "family_id": family.family_id,
                }
                for family in definitions
            ],
            "feature_axis_id": feature_axis_id,
            "prior_manifest_digest": prior.manifest_digest,
        },
        schema_version="1",
    )
    opportunity_axis_id = stable_id(
        "receiver_family_opportunity_axis",
        {
            "family_axis_id": family_axis_id,
            "receiver_axis_id": receiver_axis_id,
        },
        schema_version="1",
    )
    family_ids = tuple(family.family_id for family in definitions)
    opportunity_ids = tuple(
        (
            receiver,
            family_id,
            stable_id(
                "receiver_family_opportunity",
                {
                    "family_id": family_id,
                    "opportunity_axis_id": opportunity_axis_id,
                    "receiver": receiver,
                },
                schema_version="1",
            ),
        )
        for receiver in receivers
        for family_id in family_ids
    )
    self = object.__new__(FrozenReceiverFamilyOpportunityUniverse)
    values: dict[str, object] = {
        "receiver_ids": receivers,
        "receiver_universe_id": _nonempty_name(
            receiver_universe_id, field_name="receiver_universe_id"
        ),
        "receiver_axis_id": receiver_axis_id,
        "feature_ids": features,
        "feature_axis_id": feature_axis_id,
        "driver_ids": prior.driver_ids,
        "family_definitions": definitions,
        "family_ids": family_ids,
        "family_axis_id": family_axis_id,
        "opportunity_ids": opportunity_ids,
        "opportunity_axis_id": opportunity_axis_id,
        "prior_resource_id": prior.resource_id,
        "prior_version": prior.version,
        "prior_manifest_digest": prior.manifest_digest,
        "prior_content_id": _nonempty_name(
            prior_content_id, field_name="prior_content_id"
        ),
        "root_input_identity_id": _nonempty_name(
            root_input_identity_id, field_name="root_input_identity_id"
        ),
        "root_input_digest": _nonempty_name(
            root_input_digest, field_name="root_input_digest"
        ),
        "cosine_threshold": threshold,
        "source_policy": _RECEIVER_FAMILY_UNIVERSE_POLICY,
        "_producer_marker": _RECEIVER_FAMILY_UNIVERSE_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "universe_id",
        stable_id(
            "frozen_receiver_family_opportunity_universe",
            self._identity_payload(),
            schema_version="1",
        ),
    )
    self._require_intact()
    return self


@dataclass(frozen=True, slots=True, init=False)
class ReceiverFamilyOpportunityUniverseReuseBinding:
    """Bind a resampled root to one unchanged receiver-family opportunity axis."""

    source_universe_id: str
    child_universe_id: str
    receiver_axis_id: str
    feature_axis_id: str
    family_axis_id: str
    opportunity_axis_id: str
    family_ids: tuple[str, ...]
    prior_content_id: str
    source_root_input_identity_id: str
    source_root_input_digest: str
    child_root_input_identity_id: str
    child_root_input_digest: str
    operation: str
    plan_id: str
    resample_index: int
    materialized_input_id: str
    binding_id: str
    _source_universe: FrozenReceiverFamilyOpportunityUniverse = field(repr=False)
    _child_universe: FrozenReceiverFamilyOpportunityUniverse = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverFamilyOpportunityUniverseReuseBinding is producer-owned; "
            "use bind_reused_receiver_family_opportunity_universe()"
        )

    @classmethod
    def _from_resample(
        cls,
        source_universe: FrozenReceiverFamilyOpportunityUniverse,
        child_universe: FrozenReceiverFamilyOpportunityUniverse,
        *,
        operation: str,
        plan_id: str,
        resample_index: int,
        materialized_input_id: str,
    ) -> ReceiverFamilyOpportunityUniverseReuseBinding:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "source_universe_id": source_universe.universe_id,
            "child_universe_id": child_universe.universe_id,
            "receiver_axis_id": source_universe.receiver_axis_id,
            "feature_axis_id": source_universe.feature_axis_id,
            "family_axis_id": source_universe.family_axis_id,
            "opportunity_axis_id": source_universe.opportunity_axis_id,
            "family_ids": source_universe.family_ids,
            "prior_content_id": source_universe.prior_content_id,
            "source_root_input_identity_id": source_universe.root_input_identity_id,
            "source_root_input_digest": source_universe.root_input_digest,
            "child_root_input_identity_id": child_universe.root_input_identity_id,
            "child_root_input_digest": child_universe.root_input_digest,
            "operation": operation,
            "plan_id": plan_id,
            "resample_index": resample_index,
            "materialized_input_id": materialized_input_id,
            "_source_universe": source_universe,
            "_child_universe": child_universe,
            "_producer_marker": _RECEIVER_FAMILY_UNIVERSE_REUSE_PRODUCER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "binding_id",
            stable_id(
                "receiver_family_opportunity_universe_reuse_binding",
                self._identity_payload(),
                schema_version="1",
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_universe_id": self.source_universe_id,
            "child_universe_id": self.child_universe_id,
            "receiver_axis_id": self.receiver_axis_id,
            "feature_axis_id": self.feature_axis_id,
            "family_axis_id": self.family_axis_id,
            "opportunity_axis_id": self.opportunity_axis_id,
            "family_ids": list(self.family_ids),
            "prior_content_id": self.prior_content_id,
            "source_root_input_identity_id": self.source_root_input_identity_id,
            "source_root_input_digest": self.source_root_input_digest,
            "child_root_input_identity_id": self.child_root_input_identity_id,
            "child_root_input_digest": self.child_root_input_digest,
            "operation": self.operation,
            "plan_id": self.plan_id,
            "resample_index": self.resample_index,
            "materialized_input_id": self.materialized_input_id,
        }

    def _require_intact(self) -> None:
        try:
            source = self._source_universe
            child = self._child_universe
            if not isinstance(
                source, FrozenReceiverFamilyOpportunityUniverse
            ) or not isinstance(child, FrozenReceiverFamilyOpportunityUniverse):
                raise TypeError("invalid receiver-family opportunity universe type")
            source._require_intact()
            child._require_intact()
            operation = _nonempty_name(self.operation, field_name="operation")
            plan_id = _nonempty_name(self.plan_id, field_name="plan_id")
            materialized_input_id = _nonempty_name(
                self.materialized_input_id,
                field_name="materialized_input_id",
            )
            if (
                isinstance(self.resample_index, bool)
                or not isinstance(self.resample_index, int)
                or self.resample_index < 0
            ):
                raise ValueError("resample_index must be a non-negative integer")
            families = _canonical_names(self.family_ids, field_name="family_ids")
            expected_id = stable_id(
                "receiver_family_opportunity_universe_reuse_binding",
                self._identity_payload(),
                schema_version="1",
            )
            valid = bool(
                self._producer_marker == _RECEIVER_FAMILY_UNIVERSE_REUSE_PRODUCER
                and operation == self.operation
                and plan_id == self.plan_id
                and materialized_input_id == self.materialized_input_id
                and self.source_universe_id == source.universe_id
                and self.child_universe_id == child.universe_id
                and self.receiver_axis_id
                == source.receiver_axis_id
                == child.receiver_axis_id
                and self.feature_axis_id
                == source.feature_axis_id
                == child.feature_axis_id
                and self.family_axis_id == source.family_axis_id == child.family_axis_id
                and self.opportunity_axis_id
                == source.opportunity_axis_id
                == child.opportunity_axis_id
                and families == self.family_ids == source.family_ids == child.family_ids
                and self.prior_content_id
                == source.prior_content_id
                == child.prior_content_id
                and self.source_root_input_identity_id == source.root_input_identity_id
                and self.source_root_input_digest == source.root_input_digest
                and self.child_root_input_identity_id == child.root_input_identity_id
                and self.child_root_input_digest == child.root_input_digest
                and source.receiver_ids == child.receiver_ids
                and source.feature_ids == child.feature_ids
                and source.driver_ids == child.driver_ids
                and source.family_definitions == child.family_definitions
                and source.opportunity_ids == child.opportunity_ids
                and source.prior_resource_id == child.prior_resource_id
                and source.prior_version == child.prior_version
                and source.prior_manifest_digest == child.prior_manifest_digest
                and source.cosine_threshold == child.cosine_threshold
                and source.source_policy == child.source_policy
                and self.binding_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-family universe reuse binding failed integrity validation",
                code=(
                    "receiver_family_opportunity_universe_reuse_binding_"
                    "integrity_violation"
                ),
                field="binding_id",
                remediation=(
                    "Rebind the intact source and child family universes from the "
                    "exact materialized resampling plan"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Receiver-family universe reuse binding failed integrity validation",
                code=(
                    "receiver_family_opportunity_universe_reuse_binding_"
                    "integrity_violation"
                ),
                field="binding_id",
                remediation=(
                    "Rebind the intact source and child family universes from the "
                    "exact materialized resampling plan"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        """Return the exact source-to-child family-axis lineage."""

        self._require_intact()
        return {"binding_id": self.binding_id, **self._identity_payload()}


def bind_reused_receiver_family_opportunity_universe(
    source_universe: FrozenReceiverFamilyOpportunityUniverse,
    child_universe: FrozenReceiverFamilyOpportunityUniverse,
    *,
    operation: str,
    plan_id: str,
    resample_index: int,
    materialized_input_id: str,
) -> ReceiverFamilyOpportunityUniverseReuseBinding:
    """Bind a root-specific child to the unchanged source family opportunity axis."""

    if not isinstance(
        source_universe, FrozenReceiverFamilyOpportunityUniverse
    ) or not isinstance(child_universe, FrozenReceiverFamilyOpportunityUniverse):
        raise TypeError("source_universe and child_universe must be frozen universes")
    source_universe._require_intact()
    child_universe._require_intact()
    axes_match = bool(
        source_universe.receiver_ids == child_universe.receiver_ids
        and source_universe.receiver_axis_id == child_universe.receiver_axis_id
        and source_universe.feature_ids == child_universe.feature_ids
        and source_universe.feature_axis_id == child_universe.feature_axis_id
        and source_universe.driver_ids == child_universe.driver_ids
        and source_universe.family_definitions == child_universe.family_definitions
        and source_universe.family_ids == child_universe.family_ids
        and source_universe.family_axis_id == child_universe.family_axis_id
        and source_universe.opportunity_ids == child_universe.opportunity_ids
        and source_universe.opportunity_axis_id == child_universe.opportunity_axis_id
        and source_universe.prior_resource_id == child_universe.prior_resource_id
        and source_universe.prior_version == child_universe.prior_version
        and source_universe.prior_manifest_digest
        == child_universe.prior_manifest_digest
        and source_universe.prior_content_id == child_universe.prior_content_id
        and source_universe.cosine_threshold == child_universe.cosine_threshold
        and source_universe.source_policy == child_universe.source_policy
    )
    if not axes_match:
        raise ContractError(
            "Resampled child family opportunity axis differs from the source axis",
            code="receiver_family_opportunity_universe_reuse_axis_mismatch",
            field="receiver_axis_id,feature_axis_id,family_axis_id,opportunity_axis_id",
            remediation=(
                "Freeze the child from the complete source receiver IDs, target "
                "prior, root feature axis, and family threshold"
            ),
        )
    return ReceiverFamilyOpportunityUniverseReuseBinding._from_resample(
        source_universe,
        child_universe,
        operation=_nonempty_name(operation, field_name="operation"),
        plan_id=_nonempty_name(plan_id, field_name="plan_id"),
        resample_index=resample_index,
        materialized_input_id=_nonempty_name(
            materialized_input_id,
            field_name="materialized_input_id",
        ),
    )


def _canonical_optional_names(
    values: Sequence[str], *, field_name: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    names = tuple(
        sorted(_nonempty_name(value, field_name=field_name) for value in values)
    )
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must contain unique values")
    return names


def _canonical_communication_modes(
    values: Sequence[CommunicationMode | str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("modes must be a sequence, not a string")
    try:
        selected = tuple(CommunicationMode(value) for value in values)
    except ValueError as error:
        raise ValueError("modes contains an unsupported communication mode") from error
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("modes must be non-empty and unique")
    selected_set = set(selected)
    return tuple(mode.value for mode in CommunicationMode if mode in selected_set)


def _resource_bundle_content_identity(resource_bundle: ResourceBundle) -> str:
    return str(stable_id("resource_bundle_content", asdict(resource_bundle)))


def _target_prior_content_identity(target_prior: TargetPrior) -> str:
    return str(stable_id("target_prior_content", asdict(target_prior)))


@dataclass(frozen=True, slots=True)
class FrozenReceiverFamilyLRMappingReport:
    """Complete static ResourceBundle-to-TargetPrior mapping accounting."""

    resource_bundle_content_id: str
    target_prior_content_id: str
    mapping_policy: str
    interaction_ids: tuple[str, ...]
    mapped_interaction_ids: tuple[str, ...]
    unmapped_interaction_ids: tuple[str, ...]
    ambiguous_interactions: tuple[tuple[str, tuple[str, ...]], ...]
    report_id: str = field(init=False)

    def __post_init__(self) -> None:
        resource_content_id = _nonempty_name(
            self.resource_bundle_content_id,
            field_name="resource_bundle_content_id",
        )
        prior_content_id = _nonempty_name(
            self.target_prior_content_id,
            field_name="target_prior_content_id",
        )
        if self.mapping_policy != _RECEIVER_FAMILY_LR_MAPPING_POLICY:
            raise ValueError("mapping_policy is not the released static LR policy")
        interactions = _canonical_names(
            self.interaction_ids, field_name="interaction_ids"
        )
        mapped = _canonical_optional_names(
            self.mapped_interaction_ids,
            field_name="mapped_interaction_ids",
        )
        unmapped = _canonical_optional_names(
            self.unmapped_interaction_ids,
            field_name="unmapped_interaction_ids",
        )
        ambiguous = tuple(
            sorted(
                (
                    _nonempty_name(interaction_id, field_name="interaction_id"),
                    _canonical_names(matches, field_name="candidate_driver_ids"),
                )
                for interaction_id, matches in self.ambiguous_interactions
            )
        )
        ambiguous_ids = tuple(interaction_id for interaction_id, _ in ambiguous)
        if len(ambiguous_ids) != len(set(ambiguous_ids)):
            raise ValueError("ambiguous_interactions contains duplicate interactions")
        partition = (*mapped, *unmapped, *ambiguous_ids)
        if (
            len(partition) != len(set(partition))
            or tuple(sorted(partition)) != interactions
        ):
            raise ValueError(
                "mapped, unmapped, and ambiguous interactions must exactly partition "
                "the ResourceBundle"
            )
        payload = {
            "ambiguous_interactions": [
                {
                    "candidate_driver_ids": list(matches),
                    "interaction_id": interaction_id,
                }
                for interaction_id, matches in ambiguous
            ],
            "interaction_ids": list(interactions),
            "mapped_interaction_ids": list(mapped),
            "mapping_policy": self.mapping_policy,
            "resource_bundle_content_id": resource_content_id,
            "target_prior_content_id": prior_content_id,
            "unmapped_interaction_ids": list(unmapped),
        }
        object.__setattr__(self, "resource_bundle_content_id", resource_content_id)
        object.__setattr__(self, "target_prior_content_id", prior_content_id)
        object.__setattr__(self, "interaction_ids", interactions)
        object.__setattr__(self, "mapped_interaction_ids", mapped)
        object.__setattr__(self, "unmapped_interaction_ids", unmapped)
        object.__setattr__(self, "ambiguous_interactions", ambiguous)
        object.__setattr__(
            self,
            "report_id",
            stable_id(
                "receiver_family_lr_mapping_report",
                payload,
                schema_version="1",
            ),
        )

    @property
    def ambiguous_interaction_ids(self) -> tuple[str, ...]:
        return tuple(
            interaction_id for interaction_id, _ in self.ambiguous_interactions
        )

    def _require_intact(self) -> None:
        try:
            repeated = FrozenReceiverFamilyLRMappingReport(
                resource_bundle_content_id=self.resource_bundle_content_id,
                target_prior_content_id=self.target_prior_content_id,
                mapping_policy=self.mapping_policy,
                interaction_ids=self.interaction_ids,
                mapped_interaction_ids=self.mapped_interaction_ids,
                unmapped_interaction_ids=self.unmapped_interaction_ids,
                ambiguous_interactions=self.ambiguous_interactions,
            )
            valid = self == repeated
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen receiver-family LR mapping report failed integrity validation",
                code="frozen_receiver_family_lr_mapping_report_integrity_violation",
                field="report_id",
                remediation=(
                    "Rebuild it from the complete resource bundle and target prior"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Frozen receiver-family LR mapping report failed integrity validation",
                code="frozen_receiver_family_lr_mapping_report_integrity_violation",
                field="report_id",
                remediation=(
                    "Rebuild it from the complete resource bundle and target prior"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "report_id": self.report_id,
            "mapping_policy": self.mapping_policy,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "target_prior_content_id": self.target_prior_content_id,
            "interaction_count": len(self.interaction_ids),
            "mapped_count": len(self.mapped_interaction_ids),
            "unmapped_count": len(self.unmapped_interaction_ids),
            "ambiguous_count": len(self.ambiguous_interactions),
            "interaction_ids": list(self.interaction_ids),
            "mapped_interaction_ids": list(self.mapped_interaction_ids),
            "unmapped_interaction_ids": list(self.unmapped_interaction_ids),
            "ambiguous_interactions": [
                {
                    "interaction_id": interaction_id,
                    "candidate_driver_ids": list(matches),
                }
                for interaction_id, matches in self.ambiguous_interactions
            ],
        }


@dataclass(frozen=True, slots=True)
class FrozenReceiverFamilyLRMembership:
    """One uniquely mapped static LR-to-driver-to-family membership."""

    interaction_id: str
    molecular_lr_equivalence_id: str
    driver_id: str
    family_id: str
    membership_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "interaction_id": self.interaction_id,
            "molecular_lr_equivalence_id": self.molecular_lr_equivalence_id,
            "driver_id": self.driver_id,
            "family_id": self.family_id,
            "membership_id": self.membership_id,
        }


@dataclass(frozen=True, slots=True)
class _FrozenReceiverFamilyLRUniverseParts:
    molecular_lr_equivalence_universe: FrozenMolecularLREquivalenceUniverse
    mapping_report: FrozenReceiverFamilyLRMappingReport
    memberships: tuple[FrozenReceiverFamilyLRMembership, ...]
    membership_axis_id: str
    modes: tuple[str, ...]
    mode_axis_id: str
    hypothesis_ids: tuple[tuple[str, str, str, str, str, str], ...]
    hypothesis_axis_id: str
    opportunity_ids: tuple[tuple[str, str, str, str, str, str, str, str], ...]
    opportunity_axis_id: str


def _lr_source_mismatch(message: str, *, field_name: str) -> ContractError:
    return ContractError(
        message,
        code="frozen_receiver_family_lr_source_mismatch",
        field=field_name,
        remediation=(
            "Use the exact complete ResourceBundle and TargetPrior that own the "
            "declared content IDs and frozen family parent"
        ),
    )


def _derive_receiver_family_lr_universe(
    receiver_family_universe: FrozenReceiverFamilyOpportunityUniverse,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    resource_bundle_content_id: str,
    target_prior_content_id: str,
    modes: Sequence[CommunicationMode | str],
) -> _FrozenReceiverFamilyLRUniverseParts:
    if not isinstance(
        receiver_family_universe, FrozenReceiverFamilyOpportunityUniverse
    ):
        raise TypeError(
            "receiver_family_universe must be a FrozenReceiverFamilyOpportunityUniverse"
        )
    receiver_family_universe._require_intact()
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not resource_bundle.interactions:
        raise _lr_source_mismatch(
            "ResourceBundle contains no LR interactions",
            field_name="resource_bundle.interactions",
        )
    resource_content_id = _nonempty_name(
        resource_bundle_content_id,
        field_name="resource_bundle_content_id",
    )
    prior_content_id = _nonempty_name(
        target_prior_content_id,
        field_name="target_prior_content_id",
    )
    if resource_content_id != _resource_bundle_content_identity(resource_bundle):
        raise _lr_source_mismatch(
            "resource_bundle_content_id does not authenticate ResourceBundle content",
            field_name="resource_bundle_content_id",
        )
    if prior_content_id != _target_prior_content_identity(target_prior):
        raise _lr_source_mismatch(
            "target_prior_content_id does not authenticate TargetPrior content",
            field_name="target_prior_content_id",
        )
    if (
        target_prior.resource_id != receiver_family_universe.prior_resource_id
        or target_prior.version != receiver_family_universe.prior_version
        or target_prior.manifest_digest
        != receiver_family_universe.prior_manifest_digest
        or prior_content_id != receiver_family_universe.prior_content_id
        or target_prior.driver_ids != receiver_family_universe.driver_ids
    ):
        raise _lr_source_mismatch(
            "TargetPrior differs from the frozen strict-family parent",
            field_name="target_prior,receiver_family_universe",
        )
    if (
        resource_bundle.species is not target_prior.species
        or resource_bundle.gene_namespace is not target_prior.gene_namespace
    ):
        raise _lr_source_mismatch(
            "ResourceBundle and TargetPrior species or namespace differ",
            field_name="resource_bundle,target_prior",
        )

    family_by_driver = {
        driver_id: family.family_id
        for family in receiver_family_universe.family_definitions
        for driver_id in family.driver_ids
    }
    molecular_universe = freeze_molecular_lr_equivalence_universe(resource_bundle)
    molecular_by_interaction = {
        record.interaction_id: cast(str, record.molecular_lr_equivalence_id)
        for record in molecular_universe.mapping_records
        if record.status is MolecularLRMappingStatus.MAPPED
    }
    if (
        len(molecular_universe.source_bindings) != 1
        or molecular_universe.source_bindings[0].resource_bundle_content_id
        != resource_content_id
    ):
        raise _lr_source_mismatch(
            "Molecular LR equivalence source differs from the ResourceBundle",
            field_name="molecular_lr_equivalence_universe",
        )
    known_drivers = set(target_prior.driver_ids)
    mapped: list[tuple[str, str, str, str]] = []
    unmapped: list[str] = []
    ambiguous: list[tuple[str, tuple[str, ...]]] = []
    for interaction in resource_bundle.interactions:
        molecular_lr_equivalence_id = molecular_by_interaction.get(
            interaction.interaction_id
        )
        if molecular_lr_equivalence_id is None:
            unmapped.append(interaction.interaction_id)
            continue
        if target_prior.driver_kind == "interaction":
            candidates = {interaction.interaction_id}
        else:
            candidates = {interaction.ligand_name}
            if len(interaction.ligand_subunits) == 1:
                candidates.add(interaction.ligand_subunits[0])
        matches = tuple(sorted(candidates.intersection(known_drivers)))
        if not matches:
            unmapped.append(interaction.interaction_id)
        elif len(matches) > 1:
            ambiguous.append((interaction.interaction_id, matches))
        else:
            driver_id = matches[0]
            mapped.append(
                (
                    interaction.interaction_id,
                    molecular_lr_equivalence_id,
                    driver_id,
                    family_by_driver[driver_id],
                )
            )
    mapping_report = FrozenReceiverFamilyLRMappingReport(
        resource_bundle_content_id=resource_content_id,
        target_prior_content_id=prior_content_id,
        mapping_policy=_RECEIVER_FAMILY_LR_MAPPING_POLICY,
        interaction_ids=tuple(
            interaction.interaction_id for interaction in resource_bundle.interactions
        ),
        mapped_interaction_ids=tuple(
            interaction_id for interaction_id, _, _, _ in mapped
        ),
        unmapped_interaction_ids=tuple(unmapped),
        ambiguous_interactions=tuple(ambiguous),
    )
    if not mapped:
        raise _lr_source_mismatch(
            "ResourceBundle has no unique TargetPrior driver mapping",
            field_name="resource_bundle.interactions,target_prior.driver_ids",
        )
    mapped = sorted(mapped)
    membership_axis_id = str(
        stable_id(
            "receiver_family_lr_membership_axis",
            {
                "family_axis_id": receiver_family_universe.family_axis_id,
                "mapped_memberships": [list(item) for item in mapped],
                "mapping_policy": _RECEIVER_FAMILY_LR_MAPPING_POLICY,
                "mapping_report_id": mapping_report.report_id,
                "molecular_lr_axis_id": molecular_universe.molecular_lr_axis_id,
                "resource_bundle_content_id": resource_content_id,
                "target_prior_content_id": prior_content_id,
            },
            schema_version="2",
        )
    )
    memberships = tuple(
        FrozenReceiverFamilyLRMembership(
            interaction_id=interaction_id,
            molecular_lr_equivalence_id=molecular_lr_equivalence_id,
            driver_id=driver_id,
            family_id=family_id,
            membership_id=str(
                stable_id(
                    "receiver_family_lr_membership",
                    {
                        "driver_id": driver_id,
                        "family_id": family_id,
                        "interaction_id": interaction_id,
                        "membership_axis_id": membership_axis_id,
                        "molecular_lr_equivalence_id": (molecular_lr_equivalence_id),
                    },
                    schema_version="2",
                )
            ),
        )
        for (
            interaction_id,
            molecular_lr_equivalence_id,
            driver_id,
            family_id,
        ) in mapped
    )
    canonical_modes = _canonical_communication_modes(modes)
    mode_axis_id = str(
        stable_id(
            "communication_mode_axis",
            {"modes": list(canonical_modes)},
            schema_version="1",
        )
    )
    hypothesis_axis_id = str(
        stable_id(
            "receiver_family_lr_hypothesis_axis",
            {
                "membership_axis_id": membership_axis_id,
                "mode_axis_id": mode_axis_id,
            },
            schema_version="2",
        )
    )
    hypothesis_ids = tuple(
        (
            membership.interaction_id,
            membership.driver_id,
            membership.family_id,
            membership.membership_id,
            mode,
            str(
                stable_id(
                    "receiver_family_lr_hypothesis",
                    {
                        "hypothesis_axis_id": hypothesis_axis_id,
                        "membership_id": membership.membership_id,
                        "mode": mode,
                    },
                    schema_version="2",
                )
            ),
        )
        for membership in memberships
        for mode in canonical_modes
    )
    opportunity_axis_id = str(
        stable_id(
            "receiver_family_lr_opportunity_axis",
            {
                "hypothesis_axis_id": hypothesis_axis_id,
                "receiver_axis_id": receiver_family_universe.receiver_axis_id,
            },
            schema_version="2",
        )
    )
    opportunity_ids = tuple(
        (
            receiver,
            interaction_id,
            driver_id,
            family_id,
            membership_id,
            mode,
            hypothesis_id,
            str(
                stable_id(
                    "receiver_family_lr_opportunity",
                    {
                        "hypothesis_id": hypothesis_id,
                        "opportunity_axis_id": opportunity_axis_id,
                        "receiver": receiver,
                    },
                    schema_version="2",
                )
            ),
        )
        for receiver in receiver_family_universe.receiver_ids
        for (
            interaction_id,
            driver_id,
            family_id,
            membership_id,
            mode,
            hypothesis_id,
        ) in hypothesis_ids
    )
    return _FrozenReceiverFamilyLRUniverseParts(
        molecular_lr_equivalence_universe=molecular_universe,
        mapping_report=mapping_report,
        memberships=memberships,
        membership_axis_id=membership_axis_id,
        modes=canonical_modes,
        mode_axis_id=mode_axis_id,
        hypothesis_ids=hypothesis_ids,
        hypothesis_axis_id=hypothesis_axis_id,
        opportunity_ids=opportunity_ids,
        opportunity_axis_id=opportunity_axis_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenReceiverFamilyLRHypothesisUniverse:
    """Run-root receiver x family x LR x mode opportunities frozen pre-fit."""

    receiver_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    receiver_family_universe_id: str
    receiver_universe_id: str
    receiver_axis_id: str
    feature_axis_id: str
    family_axis_id: str
    root_input_identity_id: str
    root_input_digest: str
    resource_id: str
    resource_version: str
    resource_manifest_digest: str
    resource_bundle_content_id: str
    target_prior_resource_id: str
    target_prior_version: str
    target_prior_manifest_digest: str
    target_prior_content_id: str
    molecular_lr_equivalence_universe: FrozenMolecularLREquivalenceUniverse
    molecular_lr_equivalence_universe_id: str
    molecular_lr_axis_id: str
    mapping_policy: str
    mapping_report: FrozenReceiverFamilyLRMappingReport
    memberships: tuple[FrozenReceiverFamilyLRMembership, ...]
    membership_axis_id: str
    modes: tuple[str, ...]
    mode_axis_id: str
    hypothesis_ids: tuple[tuple[str, str, str, str, str, str], ...]
    hypothesis_axis_id: str
    opportunity_ids: tuple[tuple[str, str, str, str, str, str, str, str], ...]
    opportunity_axis_id: str
    source_policy: str
    schema_version: str
    universe_id: str
    _receiver_family_universe: FrozenReceiverFamilyOpportunityUniverse = field(
        repr=False
    )
    _resource_bundle: ResourceBundle = field(repr=False)
    _target_prior: TargetPrior = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "FrozenReceiverFamilyLRHypothesisUniverse is producer-owned; use "
            "freeze_receiver_family_lr_hypothesis_universe()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "family_axis_id": self.family_axis_id,
            "feature_axis_id": self.feature_axis_id,
            "hypothesis_axis_id": self.hypothesis_axis_id,
            "mapping_policy": self.mapping_policy,
            "mapping_report_id": self.mapping_report.report_id,
            "membership_axis_id": self.membership_axis_id,
            "molecular_lr_axis_id": self.molecular_lr_axis_id,
            "molecular_lr_equivalence_universe_id": (
                self.molecular_lr_equivalence_universe_id
            ),
            "mode_axis_id": self.mode_axis_id,
            "modes": list(self.modes),
            "opportunity_axis_id": self.opportunity_axis_id,
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_family_universe_id": self.receiver_family_universe_id,
            "receiver_universe_id": self.receiver_universe_id,
            "root_input_digest": self.root_input_digest,
            "root_input_identity_id": self.root_input_identity_id,
            "resource_bundle_content_id": self.resource_bundle_content_id,
            "resource_id": self.resource_id,
            "resource_manifest_digest": self.resource_manifest_digest,
            "resource_version": self.resource_version,
            "schema_version": self.schema_version,
            "source_policy": self.source_policy,
            "target_prior_content_id": self.target_prior_content_id,
            "target_prior_manifest_digest": self.target_prior_manifest_digest,
            "target_prior_resource_id": self.target_prior_resource_id,
            "target_prior_version": self.target_prior_version,
        }

    def _require_intact(self) -> None:
        try:
            self.mapping_report._require_intact()
            self.molecular_lr_equivalence_universe.to_dict()
            expected = _derive_receiver_family_lr_universe(
                self._receiver_family_universe,
                self._resource_bundle,
                self._target_prior,
                resource_bundle_content_id=self.resource_bundle_content_id,
                target_prior_content_id=self.target_prior_content_id,
                modes=self.modes,
            )
            expected_universe_id = stable_id(
                "frozen_receiver_family_lr_hypothesis_universe",
                self._identity_payload(),
                schema_version="2",
            )
            valid = (
                self._producer_marker == _RECEIVER_FAMILY_LR_UNIVERSE_PRODUCER
                and self.source_policy == _RECEIVER_FAMILY_LR_UNIVERSE_POLICY
                and self.schema_version == _RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION
                and self.mapping_policy == _RECEIVER_FAMILY_LR_MAPPING_POLICY
                and self.receiver_ids == self._receiver_family_universe.receiver_ids
                and self.family_ids == self._receiver_family_universe.family_ids
                and self.receiver_family_universe_id
                == self._receiver_family_universe.universe_id
                and self.receiver_universe_id
                == self._receiver_family_universe.receiver_universe_id
                and self.receiver_axis_id
                == self._receiver_family_universe.receiver_axis_id
                and self.feature_axis_id
                == self._receiver_family_universe.feature_axis_id
                and self.family_axis_id == self._receiver_family_universe.family_axis_id
                and self.root_input_identity_id
                == self._receiver_family_universe.root_input_identity_id
                and self.root_input_digest
                == self._receiver_family_universe.root_input_digest
                and self.resource_id == self._resource_bundle.resource_id
                and self.resource_version == self._resource_bundle.version
                and self.resource_manifest_digest
                == self._resource_bundle.manifest_digest
                and self.target_prior_resource_id == self._target_prior.resource_id
                and self.target_prior_version == self._target_prior.version
                and self.target_prior_manifest_digest
                == self._target_prior.manifest_digest
                and self.molecular_lr_equivalence_universe
                == expected.molecular_lr_equivalence_universe
                and self.molecular_lr_equivalence_universe_id
                == expected.molecular_lr_equivalence_universe.universe_id
                and self.molecular_lr_axis_id
                == expected.molecular_lr_equivalence_universe.molecular_lr_axis_id
                and self.mapping_report == expected.mapping_report
                and self.memberships == expected.memberships
                and self.membership_axis_id == expected.membership_axis_id
                and self.modes == expected.modes
                and self.mode_axis_id == expected.mode_axis_id
                and self.hypothesis_ids == expected.hypothesis_ids
                and self.hypothesis_axis_id == expected.hypothesis_axis_id
                and self.opportunity_ids == expected.opportunity_ids
                and self.opportunity_axis_id == expected.opportunity_axis_id
                and self.universe_id == expected_universe_id
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Frozen receiver-family LR hypothesis universe failed integrity "
                "validation",
                code=(
                    "frozen_receiver_family_lr_hypothesis_universe_integrity_violation"
                ),
                field="universe_id",
                remediation=(
                    "Refreeze it from the intact family universe, ResourceBundle, "
                    "and TargetPrior"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Frozen receiver-family LR hypothesis universe failed integrity "
                "validation",
                code=(
                    "frozen_receiver_family_lr_hypothesis_universe_integrity_violation"
                ),
                field="universe_id",
                remediation=(
                    "Refreeze it from the intact family universe, ResourceBundle, "
                    "and TargetPrior"
                ),
            )

    @property
    def unmapped_interaction_ids(self) -> tuple[str, ...]:
        self._require_intact()
        return self.mapping_report.unmapped_interaction_ids

    @property
    def ambiguous_interaction_ids(self) -> tuple[str, ...]:
        self._require_intact()
        return self.mapping_report.ambiguous_interaction_ids

    def hypothesis_id_for(
        self,
        interaction_id: str,
        mode: CommunicationMode | str,
    ) -> str:
        self._require_intact()
        key = (
            _nonempty_name(interaction_id, field_name="interaction_id"),
            CommunicationMode(mode).value,
        )
        by_key = {
            (candidate_interaction, candidate_mode): hypothesis_id
            for (
                candidate_interaction,
                _,
                _,
                _,
                candidate_mode,
                hypothesis_id,
            ) in self.hypothesis_ids
        }
        try:
            return by_key[key]
        except KeyError as error:
            raise ContractError(
                "LR-mode hypothesis is absent from the frozen universe",
                code="receiver_family_lr_hypothesis_not_in_universe",
                field="interaction_id,mode",
                remediation="Use one exact mapped LR and configured communication mode",
            ) from error

    def opportunity_id_for(
        self,
        receiver: str,
        interaction_id: str,
        mode: CommunicationMode | str,
    ) -> str:
        self._require_intact()
        key = (
            _nonempty_name(receiver, field_name="receiver"),
            _nonempty_name(interaction_id, field_name="interaction_id"),
            CommunicationMode(mode).value,
        )
        by_key = {
            (candidate_receiver, candidate_interaction, candidate_mode): opportunity_id
            for (
                candidate_receiver,
                candidate_interaction,
                _,
                _,
                _,
                candidate_mode,
                _,
                opportunity_id,
            ) in self.opportunity_ids
        }
        try:
            return by_key[key]
        except KeyError as error:
            raise ContractError(
                "Receiver-LR-mode opportunity is absent from the frozen universe",
                code="receiver_family_lr_opportunity_not_in_universe",
                field="receiver,interaction_id,mode",
                remediation=(
                    "Use one exact receiver, mapped LR, and configured "
                    "communication mode"
                ),
            ) from error

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        molecular_by_membership = {
            membership.membership_id: membership.molecular_lr_equivalence_id
            for membership in self.memberships
        }
        return {
            "universe_id": self.universe_id,
            **self._identity_payload(),
            "receiver_ids": list(self.receiver_ids),
            "family_ids": list(self.family_ids),
            "receiver_family_universe": self._receiver_family_universe.to_dict(),
            "molecular_lr_equivalence_universe": (
                self.molecular_lr_equivalence_universe.to_dict()
            ),
            "mapping_report": self.mapping_report.to_dict(),
            "membership_count": len(self.memberships),
            "memberships": [membership.to_dict() for membership in self.memberships],
            "hypothesis_count": len(self.hypothesis_ids),
            "hypotheses": [
                {
                    "interaction_id": interaction_id,
                    "driver_id": driver_id,
                    "family_id": family_id,
                    "molecular_lr_equivalence_id": molecular_by_membership[
                        membership_id
                    ],
                    "membership_id": membership_id,
                    "mode": mode,
                    "hypothesis_id": hypothesis_id,
                }
                for (
                    interaction_id,
                    driver_id,
                    family_id,
                    membership_id,
                    mode,
                    hypothesis_id,
                ) in self.hypothesis_ids
            ],
            "opportunity_count": len(self.opportunity_ids),
            "opportunities": [
                {
                    "receiver": receiver,
                    "interaction_id": interaction_id,
                    "driver_id": driver_id,
                    "family_id": family_id,
                    "molecular_lr_equivalence_id": molecular_by_membership[
                        membership_id
                    ],
                    "membership_id": membership_id,
                    "mode": mode,
                    "hypothesis_id": hypothesis_id,
                    "opportunity_id": opportunity_id,
                }
                for (
                    receiver,
                    interaction_id,
                    driver_id,
                    family_id,
                    membership_id,
                    mode,
                    hypothesis_id,
                    opportunity_id,
                ) in self.opportunity_ids
            ],
        }


def freeze_receiver_family_lr_hypothesis_universe(
    receiver_family_universe: FrozenReceiverFamilyOpportunityUniverse,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    resource_bundle_content_id: str,
    target_prior_content_id: str,
    modes: Sequence[CommunicationMode | str] = (
        CommunicationMode.STATE,
        CommunicationMode.ECOSYSTEM,
    ),
) -> FrozenReceiverFamilyLRHypothesisUniverse:
    """Freeze the complete external-resource receiver x family x LR grid."""

    parts = _derive_receiver_family_lr_universe(
        receiver_family_universe,
        resource_bundle,
        target_prior,
        resource_bundle_content_id=resource_bundle_content_id,
        target_prior_content_id=target_prior_content_id,
        modes=modes,
    )
    self = object.__new__(FrozenReceiverFamilyLRHypothesisUniverse)
    values: dict[str, object] = {
        "receiver_ids": receiver_family_universe.receiver_ids,
        "family_ids": receiver_family_universe.family_ids,
        "receiver_family_universe_id": receiver_family_universe.universe_id,
        "receiver_universe_id": receiver_family_universe.receiver_universe_id,
        "receiver_axis_id": receiver_family_universe.receiver_axis_id,
        "feature_axis_id": receiver_family_universe.feature_axis_id,
        "family_axis_id": receiver_family_universe.family_axis_id,
        "root_input_identity_id": receiver_family_universe.root_input_identity_id,
        "root_input_digest": receiver_family_universe.root_input_digest,
        "resource_id": resource_bundle.resource_id,
        "resource_version": resource_bundle.version,
        "resource_manifest_digest": resource_bundle.manifest_digest,
        "resource_bundle_content_id": resource_bundle_content_id,
        "target_prior_resource_id": target_prior.resource_id,
        "target_prior_version": target_prior.version,
        "target_prior_manifest_digest": target_prior.manifest_digest,
        "target_prior_content_id": target_prior_content_id,
        "molecular_lr_equivalence_universe": (parts.molecular_lr_equivalence_universe),
        "molecular_lr_equivalence_universe_id": (
            parts.molecular_lr_equivalence_universe.universe_id
        ),
        "molecular_lr_axis_id": (
            parts.molecular_lr_equivalence_universe.molecular_lr_axis_id
        ),
        "mapping_policy": _RECEIVER_FAMILY_LR_MAPPING_POLICY,
        "mapping_report": parts.mapping_report,
        "memberships": parts.memberships,
        "membership_axis_id": parts.membership_axis_id,
        "modes": parts.modes,
        "mode_axis_id": parts.mode_axis_id,
        "hypothesis_ids": parts.hypothesis_ids,
        "hypothesis_axis_id": parts.hypothesis_axis_id,
        "opportunity_ids": parts.opportunity_ids,
        "opportunity_axis_id": parts.opportunity_axis_id,
        "source_policy": _RECEIVER_FAMILY_LR_UNIVERSE_POLICY,
        "schema_version": _RECEIVER_FAMILY_LR_UNIVERSE_SCHEMA_VERSION,
        "_receiver_family_universe": receiver_family_universe,
        "_resource_bundle": resource_bundle,
        "_target_prior": target_prior,
        "_producer_marker": _RECEIVER_FAMILY_LR_UNIVERSE_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "universe_id",
        stable_id(
            "frozen_receiver_family_lr_hypothesis_universe",
            self._identity_payload(),
            schema_version="2",
        ),
    )
    self._require_intact()
    return self


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
    frozen_family_axis_parent_id: str | None
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
        frozen_family_axis_parent_id: str | None = None,
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
        if frozen_family_axis_parent_id is not None:
            _nonempty_name(
                frozen_family_axis_parent_id,
                field_name="frozen_family_axis_parent_id",
            )
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
            frozen_family_axis_parent_id=frozen_family_axis_parent_id,
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
            "frozen_family_axis_parent_id": frozen_family_axis_parent_id,
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
            if self.frozen_family_axis_parent_id is not None:
                _nonempty_name(
                    self.frozen_family_axis_parent_id,
                    field_name="frozen_family_axis_parent_id",
                )
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
                frozen_family_axis_parent_id=self.frozen_family_axis_parent_id,
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


@dataclass(frozen=True, slots=True, init=False)
class ReceiverFamilyAxisRefitResult:
    """Typed inner-train gate refit on one immutable outer family axis."""

    refit_id: str
    outer_parent_id: str
    outer_training_subject_ids: tuple[str, ...]
    receiver: str
    fold_id: str
    training_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    filter_universe_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    receptor_evidence_digest: str
    evidence_subject_ids: tuple[str, ...]
    evidence_row_count: int
    status: ReceiverFamilyAxisRefitStatus
    reason_code: str | None
    artifact: ReceiverFamilyTrainingArtifact | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverFamilyAxisRefitResult is producer-owned; use "
            "refit_receiver_family_training_artifact_on_frozen_axis()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "artifact_id": (
                None if self.artifact is None else self.artifact.training_artifact_id
            ),
            "evidence_row_count": self.evidence_row_count,
            "evidence_subject_ids": list(self.evidence_subject_ids),
            "family_ids": list(self.family_ids),
            "feature_ids": list(self.feature_ids),
            "filter_universe_id": self.filter_universe_id,
            "fold_id": self.fold_id,
            "method": _FROZEN_AXIS_REFIT_METHOD,
            "outer_parent_id": self.outer_parent_id,
            "outer_training_subject_ids": list(self.outer_training_subject_ids),
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "receptor_evidence_digest": self.receptor_evidence_digest,
            "status": self.status,
            "training_subject_ids": list(self.training_subject_ids),
            "validation_subject_ids": list(self.validation_subject_ids),
        }

    @classmethod
    def _from_producer(
        cls,
        *,
        outer_parent: ReceiverFamilyTrainingArtifact,
        fold_id: str,
        training_subject_ids: tuple[str, ...],
        validation_subject_ids: tuple[str, ...],
        gate_fit: _PooledReceptorGateFit,
        artifact: ReceiverFamilyTrainingArtifact | None,
        reason_code: str | None,
    ) -> ReceiverFamilyAxisRefitResult:
        self = object.__new__(cls)
        status: ReceiverFamilyAxisRefitStatus = (
            "observed" if artifact is not None else "not_estimable"
        )
        values: dict[str, object] = {
            "outer_parent_id": outer_parent.training_artifact_id,
            "outer_training_subject_ids": outer_parent.training_subject_ids,
            "receiver": outer_parent.receiver,
            "fold_id": fold_id,
            "training_subject_ids": training_subject_ids,
            "validation_subject_ids": validation_subject_ids,
            "filter_universe_id": outer_parent.filter_universe_id,
            "feature_ids": outer_parent.family_basis.feature_ids,
            "family_ids": outer_parent.family_basis.family_ids,
            "receptor_evidence_digest": gate_fit.evidence_digest,
            "evidence_subject_ids": gate_fit.evidence_subject_ids,
            "evidence_row_count": gate_fit.evidence_row_count,
            "status": status,
            "reason_code": reason_code,
            "artifact": artifact,
            "_producer_marker": _FROZEN_AXIS_REFIT_PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "refit_id",
            str(
                stable_id(
                    "receiver_family_frozen_axis_refit",
                    self._identity_payload(),
                    schema_version="1",
                )
            ),
        )
        self._require_producer_owned()
        return self

    def _require_producer_owned(self) -> None:
        try:
            outer_subjects = _subject_ids(
                self.outer_training_subject_ids,
                field_name="outer_training_subject_ids",
            )
            training_subjects = _subject_ids(
                self.training_subject_ids,
                field_name="training_subject_ids",
            )
            validation_subjects = _subject_ids(
                self.validation_subject_ids,
                field_name="validation_subject_ids",
            )
            evidence_subjects = _subject_ids(
                self.evidence_subject_ids,
                field_name="evidence_subject_ids",
                allow_empty=True,
            )
            if set(training_subjects).intersection(validation_subjects):
                raise ValueError("inner training and validation subjects overlap")
            if set(training_subjects).union(validation_subjects) != set(outer_subjects):
                raise ValueError("inner subjects do not partition the outer parent")
            if not set(evidence_subjects).issubset(training_subjects):
                raise ValueError("receptor evidence includes non-training subjects")
            if (
                isinstance(self.evidence_row_count, bool)
                or not isinstance(self.evidence_row_count, int)
                or self.evidence_row_count < len(evidence_subjects)
            ):
                raise ValueError("invalid receptor evidence row count")
            for value, field_name in (
                (self.refit_id, "refit_id"),
                (self.outer_parent_id, "outer_parent_id"),
                (self.receiver, "receiver"),
                (self.fold_id, "fold_id"),
                (self.filter_universe_id, "filter_universe_id"),
                (self.receptor_evidence_digest, "receptor_evidence_digest"),
            ):
                _nonempty_name(value, field_name=field_name)
            if not self.feature_ids or len(set(self.feature_ids)) != len(
                self.feature_ids
            ):
                raise ValueError("invalid frozen feature axis")
            if not self.family_ids or len(set(self.family_ids)) != len(self.family_ids):
                raise ValueError("invalid frozen family axis")
            observed = (
                self.status == "observed"
                and self.reason_code is None
                and self.artifact is not None
                and len(training_subjects) >= 2
                and len(evidence_subjects) >= 2
            )
            not_estimable = (
                self.status == "not_estimable"
                and self.artifact is None
                and self.reason_code
                in {
                    _FROZEN_AXIS_INSUFFICIENT_TRAINING_SUBJECTS,
                    _FROZEN_AXIS_INSUFFICIENT_RECEIVER_SUBJECTS,
                }
                and (
                    (
                        self.reason_code == _FROZEN_AXIS_INSUFFICIENT_TRAINING_SUBJECTS
                        and len(training_subjects) < 2
                    )
                    or (
                        self.reason_code == _FROZEN_AXIS_INSUFFICIENT_RECEIVER_SUBJECTS
                        and len(training_subjects) >= 2
                        and len(evidence_subjects) < 2
                    )
                )
            )
            if not (observed or not_estimable):
                raise ValueError("invalid frozen-axis refit state")
            if self.artifact is not None:
                self.artifact._require_producer_owned()
                if (
                    self.artifact.frozen_family_axis_parent_id != self.outer_parent_id
                    or self.artifact.receiver != self.receiver
                    or self.artifact.fold_id != self.fold_id
                    or self.artifact.training_subject_ids != training_subjects
                    or self.artifact.filter_universe_id != self.filter_universe_id
                    or self.artifact.receptor_evidence_digest
                    != self.receptor_evidence_digest
                    or self.artifact.family_basis.feature_ids != self.feature_ids
                    or self.artifact.family_basis.family_ids != self.family_ids
                ):
                    raise ValueError("refitted artifact does not match its lineage")
            expected_id = str(
                stable_id(
                    "receiver_family_frozen_axis_refit",
                    self._identity_payload(),
                    schema_version="1",
                )
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen-axis receiver-family refit failed integrity validation",
                code="receiver_family_frozen_axis_refit_integrity_violation",
                field="refit_id",
                remediation="Repeat the refit from intact outer and inner parents",
            ) from error
        if (
            self._producer_marker != _FROZEN_AXIS_REFIT_PRODUCER_MARKER
            or expected_id != self.refit_id
        ):
            raise ContractError(
                "Frozen-axis receiver-family refit failed integrity validation",
                code="receiver_family_frozen_axis_refit_integrity_violation",
                field="refit_id",
                remediation="Repeat the refit from intact outer and inner parents",
            )

    def to_dict(self) -> dict[str, object]:
        """Return compact lineage without serializing sparse basis payloads."""

        self._require_producer_owned()
        return {"refit_id": self.refit_id, **self._identity_payload()}


def _require_exact_frozen_profile_axis(
    outer: GatedTargetBasis, inner: GatedTargetBasis
) -> None:
    same_axis = (
        inner.feature_ids == outer.feature_ids
        and inner.driver_ids == outer.driver_ids
        and inner.prior_resource_id == outer.prior_resource_id
        and inner.prior_version == outer.prior_version
        and inner.normalized_profiles.shape == outer.normalized_profiles.shape
        and np.array_equal(
            inner.normalized_profiles.indptr, outer.normalized_profiles.indptr
        )
        and np.array_equal(
            inner.normalized_profiles.indices, outer.normalized_profiles.indices
        )
        and np.array_equal(
            inner.normalized_profiles.data, outer.normalized_profiles.data
        )
        and np.array_equal(inner.pre_normalization_norms, outer.pre_normalization_norms)
    )
    if not same_axis:
        raise ContractError(
            "Inner target-prior profiles differ from the frozen outer axis",
            code="receiver_family_frozen_axis_parent_mismatch",
            field="prior,feature_ids",
            remediation="Use the exact prior and feature axis from the outer parent",
        )


def refit_receiver_family_training_artifact_on_frozen_axis(
    availability: BatchAvailability,
    prior: TargetPrior,
    outer_parent: ReceiverFamilyTrainingArtifact,
    *,
    fold_id: str,
    inner_training_subject_ids: Sequence[str],
    inner_validation_subject_ids: Sequence[str],
) -> ReceiverFamilyAxisRefitResult:
    """Refit inner-training receptor gates without relearning any family axis.

    ``availability`` must be an application of the outer frozen interaction
    universe to the physical inner-training subjects.  Validation values are
    deliberately absent from this API; validation subject IDs are accepted only
    to prove that the inner split exactly partitions the outer parent subjects.
    """

    if not isinstance(availability, BatchAvailability):
        raise TypeError("availability must be a BatchAvailability")
    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    if not isinstance(outer_parent, ReceiverFamilyTrainingArtifact):
        raise TypeError("outer_parent must be a ReceiverFamilyTrainingArtifact")
    outer_parent._require_producer_owned()
    availability.frozen_interaction_universe._require_intact()
    inner_fold = _nonempty_name(fold_id, field_name="fold_id")
    training_subjects = _subject_ids(
        inner_training_subject_ids,
        field_name="inner_training_subject_ids",
    )
    validation_subjects = _subject_ids(
        inner_validation_subject_ids,
        field_name="inner_validation_subject_ids",
    )
    if set(training_subjects).intersection(validation_subjects):
        raise ContractError(
            "Inner receiver-family training and validation subjects overlap",
            code="receiver_family_frozen_axis_subject_leakage",
            field="inner_training_subject_ids,inner_validation_subject_ids",
            remediation="Use a disjoint subject-blocked inner partition",
        )
    if set(training_subjects).union(validation_subjects) != set(
        outer_parent.training_subject_ids
    ):
        raise ContractError(
            "Inner subjects do not exactly partition the outer family parent",
            code="receiver_family_frozen_axis_parent_scope_mismatch",
            field="inner_training_subject_ids,inner_validation_subject_ids",
            remediation="Partition every outer-training subject exactly once",
        )
    if availability.application_subject_ids != training_subjects:
        raise ContractError(
            "Inner availability subjects differ from the declared training split",
            code="receiver_family_frozen_axis_subject_leakage",
            field="availability.application_subject_ids",
            remediation="Recompute availability on only the inner-training subjects",
        )
    table_subjects = set(
        availability.sample_interactions.get("subject_id", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
    )
    if not table_subjects.issubset(training_subjects):
        raise ContractError(
            "Inner availability contains non-training subject rows",
            code="receiver_family_frozen_axis_subject_leakage",
            field="availability.sample_interactions.subject_id",
            remediation="Physically subset availability to inner-training subjects",
        )
    if (
        availability.filter_application
        is not InteractionFilterApplication.FROZEN_APPLICATION_V1
        or availability.filter_universe_id != outer_parent.filter_universe_id
        or outer_parent.frozen_family_axis_parent_id is not None
    ):
        raise ContractError(
            "Inner availability or parent does not use the root outer frozen universe",
            code="receiver_family_frozen_axis_parent_mismatch",
            field="filter_universe_id,outer_parent",
            remediation="Apply the root outer interaction universe to inner training",
        )
    if (
        prior.manifest_digest != outer_parent.prior_manifest_digest
        or prior.driver_ids != outer_parent.source_basis.driver_ids
    ):
        raise ContractError(
            "Target prior differs from the frozen outer driver axis",
            code="receiver_family_frozen_axis_parent_mismatch",
            field="prior",
            remediation="Use the exact TargetPrior that produced the outer parent",
        )
    if (
        outer_parent.source_basis.gate_policy
        is not ReceptorGatePolicy.HARD_ELIGIBILITY_V2
        or outer_parent.source_basis.receptor_gate_threshold is None
    ):
        raise ContractError(
            "Outer family parent does not use hard receptor eligibility",
            code="receiver_family_frozen_axis_parent_mismatch",
            field="outer_parent.source_basis.gate_policy",
            remediation="Fit the outer parent with the released hard-gate policy",
        )

    gate_fit = _pooled_receptor_gates(
        availability,
        receiver=outer_parent.receiver,
        driver_ids=outer_parent.source_basis.driver_ids,
        driver_by_interaction=outer_parent.driver_by_interaction,
    )
    reason_code: str | None = None
    if len(training_subjects) < 2:
        reason_code = _FROZEN_AXIS_INSUFFICIENT_TRAINING_SUBJECTS
    elif len(gate_fit.evidence_subject_ids) < 2:
        reason_code = _FROZEN_AXIS_INSUFFICIENT_RECEIVER_SUBJECTS
    if reason_code is not None:
        return ReceiverFamilyAxisRefitResult._from_producer(
            outer_parent=outer_parent,
            fold_id=inner_fold,
            training_subject_ids=training_subjects,
            validation_subject_ids=validation_subjects,
            gate_fit=gate_fit,
            artifact=None,
            reason_code=reason_code,
        )

    source_basis = build_gated_target_basis(
        prior,
        outer_parent.source_basis.feature_ids,
        dict(gate_fit.receptor_gates),
        gate_policy=outer_parent.source_basis.gate_policy,
        receptor_gate_threshold=outer_parent.source_basis.receptor_gate_threshold,
    )
    _require_exact_frozen_profile_axis(outer_parent.source_basis, source_basis)
    family_basis = build_family_first_basis(
        source_basis,
        outer_parent.family_basis.family_definitions,
        strict_cosine_threshold=outer_parent.family_basis.strict_cosine_threshold,
        method=outer_parent.family_basis.method,
    )
    if (
        family_basis.feature_ids != outer_parent.family_basis.feature_ids
        or family_basis.family_ids != outer_parent.family_basis.family_ids
        or family_basis.family_definitions
        != outer_parent.family_basis.family_definitions
        or family_basis.medoid_driver_ids != outer_parent.family_basis.medoid_driver_ids
    ):
        raise ContractError(
            "Inner family definitions differ from the frozen outer family axis",
            code="receiver_family_frozen_axis_parent_mismatch",
            field="family_basis",
            remediation="Reuse the exact outer family definitions and medoids",
        )
    artifact = ReceiverFamilyTrainingArtifact._from_training(
        receiver=outer_parent.receiver,
        fold_id=inner_fold,
        training_subject_ids=training_subjects,
        filter_universe_id=outer_parent.filter_universe_id,
        prior_manifest_digest=outer_parent.prior_manifest_digest,
        driver_by_interaction=outer_parent.driver_by_interaction,
        receptor_gates=gate_fit.receptor_gates,
        receptor_evidence_digest=gate_fit.evidence_digest,
        source_basis=source_basis,
        family_basis=family_basis,
        frozen_family_axis_parent_id=outer_parent.training_artifact_id,
    )
    return ReceiverFamilyAxisRefitResult._from_producer(
        outer_parent=outer_parent,
        fold_id=inner_fold,
        training_subject_ids=training_subjects,
        validation_subject_ids=validation_subjects,
        gate_fit=gate_fit,
        artifact=artifact,
        reason_code=None,
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
        gate_fit = _pooled_receptor_gates(
            availability,
            receiver=normalized_receiver,
            driver_ids=prior.driver_ids,
            driver_by_interaction=mapping,
        )
        source_basis = build_gated_target_basis(
            prior,
            feature_ids,
            dict(gate_fit.receptor_gates),
            gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
            receptor_gate_threshold=threshold,
        )
        receiver_inputs.append(
            (
                normalized_receiver,
                gate_fit.receptor_gates,
                gate_fit.evidence_digest,
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
    "FrozenReceiverFamilyLRHypothesisUniverse",
    "FrozenReceiverFamilyLRMappingReport",
    "FrozenReceiverFamilyLRMembership",
    "FrozenReceiverFamilyOpportunityUniverse",
    "ReceiverFamilyAxisRefitResult",
    "ReceiverFamilyAxisRefitStatus",
    "ReceiverFamilyOpportunityUniverseReuseBinding",
    "ReceiverFamilyTrainingArtifact",
    "bind_reused_receiver_family_opportunity_universe",
    "fit_receiver_family_training_artifact",
    "fit_receiver_family_training_artifacts",
    "freeze_receiver_family_lr_hypothesis_universe",
    "freeze_receiver_family_opportunity_universe",
    "refit_receiver_family_training_artifact_on_frozen_axis",
]
