"""Common-scale OOF target-program scores for explicit directional pairs.

This producer is deliberately source-agnostic.  It estimates one signed gene
effect from held-out receiver expression on the shared ``log1p_cpm`` scale,
splits that effect into positive and negative activation-compatible channels,
and compares each channel with a prior-only frozen target program.  The reverse
channel is attenuation-compatible evidence, not evidence of active inhibition.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Final, cast

import numpy as np
import pandas as pd
from scipy import sparse

from crychic.attribution import (
    DirectionalContrastPairSpec,
    build_gated_target_basis,
)
from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import node_context_fields
from crychic.resources import TargetPrior

from .crossfit import CrossFitArtifacts
from .receiver_universe import (
    ReceiverTrainingSupportRecord,
    ReceiverTrainingSupportStatus,
)

_SCHEMA_VERSION: Final = "1"
_SPEC_SCHEMA_VERSION: Final = "1.0.0"
_UNIVERSE_MARKER: Final = (
    "crychic.workflow.frozen_directional_target_program_universe.v1"
)
_EFFECT_MARKER: Final = "crychic.workflow.directional_target_program_effect.v1"
_COLLECTION_MARKER: Final = (
    "crychic.workflow.directional_target_program_score_collection.v1"
)
_VALUE_SCALE: Final = "log1p_cpm"
_EFFECT_BACKEND: Final = "subject_equal_multivariate_wls_context_plus_outer_fold_v1"
_SCORE_METHOD: Final = (
    "cosine_nonnegative_target_profile_vs_signed_oof_gene_effect_channel_v1"
)
_PROGRAM_POLICY: Final = "all_target_prior_drivers_prior_only_v1"
_ANALYSIS_TRACK: Final = "signed_target_program"
_SOURCE_AGNOSTIC_SENDER: Final = "__source_agnostic__"
_STATUS_OBSERVED: Final = "observed"
_STATUS_NOT_ESTIMABLE: Final = "not_estimable"
_RECEIVER_ABSENT_REASON: Final = "receiver_absent_in_outer_training"
_CHANNELS: Final = (
    "increased_activation_compatible",
    "reduced_activation_compatible",
)
_SCORE_COLUMNS: Final = (
    "crossfit_id",
    "receiver_universe_id",
    "receiver_axis_id",
    "receiver_training_support_ids",
    "pair_spec_id",
    "target_program_universe_id",
    "score_spec_id",
    "analysis_track",
    "sender",
    "receiver",
    "program_id",
    "channel",
    "score",
    "score_direction",
    "status",
    "reason_code",
    "matched_target_count",
    "effect_result_id",
    "effect_backend",
    "score_method",
    "value_scale",
    "supports_active_inhibition_claim",
    "formal_inference_allowed",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_name(value, field_name=field_name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return result


def _optional_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_name(value, field_name=field_name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be unique")
    return result


def _immutable_vector(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    canonical[canonical == 0.0] = 0.0
    canonical[np.isnan(canonical)] = np.nan
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    canonical[canonical == 0.0] = 0.0
    canonical[np.isnan(canonical)] = np.nan
    digest = hashlib.sha256()
    digest.update(canonical_json(list(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_csc(values: sparse.spmatrix) -> sparse.csc_matrix:
    result = values.astype("<f8", copy=True).tocsc()
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    result.data[result.data == 0.0] = 0.0
    result.data.setflags(write=False)
    result.indices.setflags(write=False)
    result.indptr.setflags(write=False)
    return result


def _sparse_digest(values: sparse.spmatrix) -> str:
    canonical = values.astype("<f8", copy=True).tocsc()
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.eliminate_zeros()
    canonical.data[canonical.data == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(canonical_json(list(canonical.shape)).encode("ascii"))
    digest.update(np.asarray(canonical.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _target_prior_content_id(prior: TargetPrior) -> str:
    result: str = stable_id("target_prior_content", asdict(prior))
    return result


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


@dataclass(frozen=True, slots=True, init=False)
class FrozenDirectionalTargetProgramUniverse:
    """Prior-only program universe aligned to one immutable feature order."""

    universe_id: str
    feature_ids: tuple[str, ...]
    program_ids: tuple[str, ...]
    target_prior_content_id: str
    target_prior_resource_id: str
    target_prior_version: str
    target_prior_manifest_digest: str
    target_prior_driver_kind: str
    target_prior_direction: int
    selection_policy: str
    normalized_profile_digest: str
    matched_target_counts: tuple[int, ...]
    _normalized_profiles: sparse.csc_matrix
    _target_prior: TargetPrior
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenDirectionalTargetProgramUniverse is producer-owned; use "
            "freeze_directional_target_program_universe()"
        )

    @classmethod
    def _from_prior(
        cls,
        prior: TargetPrior,
        feature_ids: tuple[str, ...],
    ) -> FrozenDirectionalTargetProgramUniverse:
        basis = build_gated_target_basis(
            prior,
            feature_ids,
            {driver: 1.0 for driver in prior.driver_ids},
        )
        profiles = _readonly_csc(basis.normalized_profiles)
        profile_digest = _sparse_digest(profiles)
        matched_counts = tuple(int(value) for value in np.diff(profiles.indptr))
        payload = {
            "feature_ids": list(feature_ids),
            "matched_target_counts": list(matched_counts),
            "normalized_profile_digest": profile_digest,
            "program_ids": list(prior.driver_ids),
            "selection_policy": _PROGRAM_POLICY,
            "target_prior_content_id": _target_prior_content_id(prior),
            "target_prior_direction": prior.direction,
            "target_prior_driver_kind": prior.driver_kind,
            "target_prior_manifest_digest": prior.manifest_digest,
            "target_prior_resource_id": prior.resource_id,
            "target_prior_version": prior.version,
        }
        self = object.__new__(cls)
        values: dict[str, object] = {
            "feature_ids": feature_ids,
            "program_ids": prior.driver_ids,
            "target_prior_content_id": payload["target_prior_content_id"],
            "target_prior_resource_id": prior.resource_id,
            "target_prior_version": prior.version,
            "target_prior_manifest_digest": prior.manifest_digest,
            "target_prior_driver_kind": prior.driver_kind,
            "target_prior_direction": prior.direction,
            "selection_policy": _PROGRAM_POLICY,
            "normalized_profile_digest": profile_digest,
            "matched_target_counts": matched_counts,
            "_normalized_profiles": profiles,
            "_target_prior": prior,
            "_producer_marker": _UNIVERSE_MARKER,
            "universe_id": stable_id(
                "directional_target_program_universe",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "feature_ids": list(self.feature_ids),
            "matched_target_counts": list(self.matched_target_counts),
            "normalized_profile_digest": self.normalized_profile_digest,
            "program_ids": list(self.program_ids),
            "selection_policy": self.selection_policy,
            "target_prior_content_id": self.target_prior_content_id,
            "target_prior_direction": self.target_prior_direction,
            "target_prior_driver_kind": self.target_prior_driver_kind,
            "target_prior_manifest_digest": self.target_prior_manifest_digest,
            "target_prior_resource_id": self.target_prior_resource_id,
            "target_prior_version": self.target_prior_version,
        }

    def _require_intact(self) -> None:
        try:
            if not isinstance(self._target_prior, TargetPrior):
                raise TypeError("target prior parent is invalid")
            features = _names(self.feature_ids, field_name="feature_ids")
            if features != self.feature_ids:
                raise ValueError("feature order changed")
            expected = FrozenDirectionalTargetProgramUniverse._from_prior(
                self._target_prior,
                features,
            )
            valid = (
                self._producer_marker == _UNIVERSE_MARKER
                and self._identity_payload() == expected._identity_payload()
                and self.universe_id == expected.universe_id
                and _sparse_digest(self._normalized_profiles)
                == self.normalized_profile_digest
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Frozen directional target-program universe failed integrity "
                "validation",
                code="directional_target_program_universe_integrity_violation",
                field="universe_id",
                remediation="Refreeze the program universe from the intact TargetPrior",
            ) from error
        if not valid:
            raise _contract_error(
                "Frozen directional target-program universe failed integrity "
                "validation",
                code="directional_target_program_universe_integrity_violation",
                field="universe_id",
                remediation="Refreeze the program universe from the intact TargetPrior",
            )

    @property
    def normalized_profiles(self) -> sparse.csc_matrix:
        """Return a defensive copy of the feature-by-program profiles."""

        self._require_intact()
        return self._normalized_profiles.copy()

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"universe_id": self.universe_id, **self._identity_payload()}


def freeze_directional_target_program_universe(
    prior: TargetPrior,
    feature_ids: Sequence[str],
) -> FrozenDirectionalTargetProgramUniverse:
    """Freeze all nonnegative TargetPrior drivers without using outcome values."""

    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    if prior.direction != 1:
        raise _contract_error(
            "Directional target programs require a nonnegative activation prior",
            code="directional_target_program_signed_prior_unsupported",
            field="direction",
            remediation=(
                "Use a positive TargetPrior or a reviewed signed-prior workflow"
            ),
        )
    features = _names(feature_ids, field_name="feature_ids")
    return FrozenDirectionalTargetProgramUniverse._from_prior(prior, features)


def freeze_crossfit_directional_target_program_universe(
    artifacts: CrossFitArtifacts,
) -> FrozenDirectionalTargetProgramUniverse:
    """Freeze the exact prior/feature universe registered by one cross-fit."""

    if not isinstance(artifacts, CrossFitArtifacts):
        raise TypeError("artifacts must be CrossFitArtifacts")
    artifacts._require_intact()
    if not artifacts.folds or any(
        not fold.receiver_responses for fold in artifacts.folds
    ):
        raise _contract_error(
            "Cross-fit has no receiver response feature universe",
            code="directional_target_program_response_registry_empty",
            field="receiver_responses",
            remediation="Run the complete receiver response stage before freezing",
        )
    feature_universes = {
        response.feature_ids
        for fold in artifacts.folds
        for response in fold.receiver_responses
    }
    prior_content_ids = {
        fold.training.target_prior_content_id for fold in artifacts.folds
    }
    if len(feature_universes) != 1 or len(prior_content_ids) != 1:
        raise _contract_error(
            "Cross-fit folds do not share one target prior and feature universe",
            code="directional_target_program_crossfit_universe_mismatch",
            field="target_prior_content_id,feature_ids",
            remediation="Use one intact cross-fit produced from a common input axis",
        )
    prior = artifacts.folds[0].training.target_prior
    if _target_prior_content_id(prior) != next(iter(prior_content_ids)):
        raise _contract_error(
            "Cross-fit TargetPrior content identity is inconsistent",
            code="directional_target_program_prior_mismatch",
            field="target_prior_content_id",
            remediation="Use the intact TargetPrior parent from cross-fit",
        )
    return freeze_directional_target_program_universe(
        prior,
        next(iter(feature_universes)),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalTargetProgramScoreSpec:
    """Pre-registered numerical policy for common-scale program recovery."""

    minimum_matched_targets: int = 1
    maximum_condition_number: float = 1.0e10
    schema_version: str = _SPEC_SCHEMA_VERSION
    effect_backend: str = field(init=False)
    score_method: str = field(init=False)
    value_scale: str = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        minimum = self.minimum_matched_targets
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("minimum_matched_targets must be an integer >= 1")
        maximum = _finite(
            self.maximum_condition_number,
            field_name="maximum_condition_number",
        )
        if maximum <= 1.0:
            raise ValueError("maximum_condition_number must be > 1")
        if self.schema_version != _SPEC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SPEC_SCHEMA_VERSION!r}")
        object.__setattr__(self, "maximum_condition_number", maximum)
        object.__setattr__(self, "effect_backend", _EFFECT_BACKEND)
        object.__setattr__(self, "score_method", _SCORE_METHOD)
        object.__setattr__(self, "value_scale", _VALUE_SCALE)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "directional_target_program_score_spec",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "effect_backend": self.effect_backend,
            "maximum_condition_number": self.maximum_condition_number,
            "minimum_matched_targets": self.minimum_matched_targets,
            "schema_version": self.schema_version,
            "score_method": self.score_method,
            "value_scale": self.value_scale,
        }

    def _require_intact(self) -> None:
        try:
            repeated = DirectionalTargetProgramScoreSpec(
                minimum_matched_targets=self.minimum_matched_targets,
                maximum_condition_number=self.maximum_condition_number,
                schema_version=self.schema_version,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _contract_error(
                "Directional target-program score specification is not intact",
                code="directional_target_program_score_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the immutable score specification",
            ) from error
        if not valid:
            raise _contract_error(
                "Directional target-program score specification is not intact",
                code="directional_target_program_score_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the immutable score specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, init=False)
class DirectionalTargetProgramEffect:
    """One receiver's signed common-scale OOF gene-effect vector."""

    result_id: str
    receiver: str
    feature_ids: tuple[str, ...]
    pair_spec_id: str
    score_spec_id: str
    receiver_universe_id: str
    receiver_axis_id: str
    fold_ids: tuple[str, ...]
    receiver_training_support_ids: tuple[str, ...]
    response_application_fold_ids: tuple[str, ...]
    forward_response_application_ids: tuple[str, ...]
    reverse_response_application_ids: tuple[str, ...]
    context_weights: tuple[tuple[str, float], ...]
    source_row_manifest_digest: str
    source_values_digest: str
    gene_effect_digest: str
    gene_effect: np.ndarray
    n_samples: int
    n_subject_context_rows: int
    n_subjects: int
    n_folds: int
    n_response_application_folds: int
    design_rank: int
    design_condition_number: float | None
    status: str
    reason_code: str | None
    effect_backend: str
    value_scale: str
    formal_inference_allowed: bool
    _receiver_training_support: tuple[ReceiverTrainingSupportRecord, ...]
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "DirectionalTargetProgramEffect is producer-owned; use "
            "score_crossfit_directional_target_programs()"
        )

    @classmethod
    def _from_fit(
        cls,
        *,
        receiver: str,
        feature_ids: tuple[str, ...],
        pair_spec_id: str,
        score_spec_id: str,
        receiver_universe_id: str,
        receiver_axis_id: str,
        fold_ids: tuple[str, ...],
        receiver_training_support: tuple[ReceiverTrainingSupportRecord, ...],
        response_application_fold_ids: tuple[str, ...],
        forward_response_application_ids: tuple[str, ...],
        reverse_response_application_ids: tuple[str, ...],
        context_weights: tuple[tuple[str, float], ...],
        source_row_manifest_digest: str,
        source_values_digest: str,
        gene_effect: np.ndarray,
        n_samples: int,
        n_subject_context_rows: int,
        n_subjects: int,
        design_rank: int,
        design_condition_number: float | None,
        status: str,
        reason_code: str | None,
    ) -> DirectionalTargetProgramEffect:
        effect = _immutable_vector(gene_effect)
        self = object.__new__(cls)
        values: dict[str, object] = {
            "receiver": receiver,
            "feature_ids": feature_ids,
            "pair_spec_id": pair_spec_id,
            "score_spec_id": score_spec_id,
            "receiver_universe_id": receiver_universe_id,
            "receiver_axis_id": receiver_axis_id,
            "fold_ids": fold_ids,
            "receiver_training_support_ids": tuple(
                record.support_record_id for record in receiver_training_support
            ),
            "response_application_fold_ids": response_application_fold_ids,
            "forward_response_application_ids": (forward_response_application_ids),
            "reverse_response_application_ids": (reverse_response_application_ids),
            "context_weights": context_weights,
            "source_row_manifest_digest": source_row_manifest_digest,
            "source_values_digest": source_values_digest,
            "gene_effect_digest": _array_digest(effect),
            "gene_effect": effect,
            "n_samples": n_samples,
            "n_subject_context_rows": n_subject_context_rows,
            "n_subjects": n_subjects,
            "n_folds": len(fold_ids),
            "n_response_application_folds": len(response_application_fold_ids),
            "design_rank": design_rank,
            "design_condition_number": design_condition_number,
            "status": status,
            "reason_code": reason_code,
            "effect_backend": _EFFECT_BACKEND,
            "value_scale": _VALUE_SCALE,
            "formal_inference_allowed": False,
            "_receiver_training_support": receiver_training_support,
            "_producer_marker": _EFFECT_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "directional_target_program_effect",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "context_weights": [list(item) for item in self.context_weights],
            "design_condition_number": self.design_condition_number,
            "design_rank": self.design_rank,
            "effect_backend": self.effect_backend,
            "feature_ids": list(self.feature_ids),
            "fold_ids": list(self.fold_ids),
            "formal_inference_allowed": self.formal_inference_allowed,
            "forward_response_application_ids": list(
                self.forward_response_application_ids
            ),
            "gene_effect_digest": self.gene_effect_digest,
            "n_folds": self.n_folds,
            "n_response_application_folds": self.n_response_application_folds,
            "n_samples": self.n_samples,
            "n_subject_context_rows": self.n_subject_context_rows,
            "n_subjects": self.n_subjects,
            "pair_spec_id": self.pair_spec_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_training_support_ids": list(self.receiver_training_support_ids),
            "receiver_universe_id": self.receiver_universe_id,
            "response_application_fold_ids": list(self.response_application_fold_ids),
            "reverse_response_application_ids": list(
                self.reverse_response_application_ids
            ),
            "score_spec_id": self.score_spec_id,
            "source_row_manifest_digest": self.source_row_manifest_digest,
            "source_values_digest": self.source_values_digest,
            "status": self.status,
            "value_scale": self.value_scale,
        }

    def _require_intact(self) -> None:
        try:
            _name(self.receiver, field_name="receiver")
            _name(self.receiver_universe_id, field_name="receiver_universe_id")
            _name(self.receiver_axis_id, field_name="receiver_axis_id")
            features = _names(self.feature_ids, field_name="feature_ids")
            folds = _names(self.fold_ids, field_name="fold_ids")
            if folds != tuple(sorted(folds)):
                raise ValueError("effect folds must use canonical order")
            application_folds = _optional_names(
                self.response_application_fold_ids,
                field_name="response_application_fold_ids",
            )
            forward_ids = _optional_names(
                self.forward_response_application_ids,
                field_name="forward_response_application_ids",
            )
            reverse_ids = _optional_names(
                self.reverse_response_application_ids,
                field_name="reverse_response_application_ids",
            )
            if not (
                len(application_folds)
                == len(forward_ids)
                == len(reverse_ids)
                == self.n_response_application_folds
            ):
                raise ValueError("response-application fold lineage differs")
            support_records = tuple(self._receiver_training_support)
            if len(support_records) != len(folds):
                raise ValueError("effect requires one support parent per fold")
            for record in support_records:
                if not isinstance(record, ReceiverTrainingSupportRecord):
                    raise TypeError("effect support parents have invalid type")
                record._require_intact()
            support_ids = tuple(record.support_record_id for record in support_records)
            support_folds = tuple(record.outer_fold_id for record in support_records)
            observed_support_folds = tuple(
                record.outer_fold_id
                for record in support_records
                if record.status is ReceiverTrainingSupportStatus.OBSERVED
            )
            if (
                support_ids != self.receiver_training_support_ids
                or support_folds != folds
                or application_folds != observed_support_folds
                or any(
                    record.receiver_id != self.receiver
                    or record.receiver_universe_id != self.receiver_universe_id
                    or record._universe.receiver_axis_id != self.receiver_axis_id
                    for record in support_records
                )
            ):
                raise ValueError("receiver support lineage differs from effect scope")
            has_absent_training = any(
                record.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE
                for record in support_records
            )
            if (
                self.gene_effect.shape != (len(features),)
                or self.gene_effect.flags.writeable
            ):
                raise ValueError("gene effect is not an immutable aligned vector")
            if self.status not in {_STATUS_OBSERVED, _STATUS_NOT_ESTIMABLE}:
                raise ValueError("effect status is invalid")
            if (self.status == _STATUS_OBSERVED) == (self.reason_code is not None):
                raise ValueError("effect status and reason are inconsistent")
            if self.status == _STATUS_OBSERVED:
                if np.any(~np.isfinite(self.gene_effect)):
                    raise ValueError("observed effect must be finite")
            elif not np.isnan(self.gene_effect).all():
                raise ValueError("not-estimable effect must remain entirely missing")
            if has_absent_training and (
                self.status != _STATUS_NOT_ESTIMABLE
                or self.reason_code != _RECEIVER_ABSENT_REASON
            ):
                raise ValueError("training-absent receiver must remain typed NE")
            counts = (
                self.n_samples,
                self.n_subject_context_rows,
                self.n_subjects,
                self.n_folds,
                self.n_response_application_folds,
                self.design_rank,
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            ):
                raise ValueError("effect counts must be non-negative integers")
            if self.n_folds != len(folds) or self.n_folds < 1:
                raise ValueError("effect requires at least one OOF fold")
            condition = self.design_condition_number
            if condition is not None and (
                not math.isfinite(condition) or condition < 1.0
            ):
                raise ValueError("design condition number is invalid")
            expected_id = stable_id(
                "directional_target_program_effect",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _EFFECT_MARKER
                and self.effect_backend == _EFFECT_BACKEND
                and self.value_scale == _VALUE_SCALE
                and self.formal_inference_allowed is False
                and _array_digest(self.gene_effect) == self.gene_effect_digest
                and expected_id == self.result_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Directional target-program effect failed integrity validation",
                code="directional_target_program_effect_integrity_violation",
                field="result_id",
                remediation="Recompute the effect from intact cross-fit applications",
            ) from error
        if not valid:
            raise _contract_error(
                "Directional target-program effect failed integrity validation",
                code="directional_target_program_effect_integrity_violation",
                field="result_id",
                remediation="Recompute the effect from intact cross-fit applications",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "result_id": self.result_id,
            **self._identity_payload(),
            "receiver_training_support": [
                record.to_dict() for record in self._receiver_training_support
            ],
        }


@dataclass(frozen=True, slots=True)
class _FoldRawView:
    fold_id: str
    forward_application_id: str
    reverse_application_id: str
    context_weights: tuple[tuple[str, float], ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_context_ids: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class _ReceiverRawRegistry:
    support_records: tuple[ReceiverTrainingSupportRecord, ...]
    views: tuple[_FoldRawView, ...]
    context_weights: tuple[tuple[str, float], ...]


def _directional_pair(
    artifacts: CrossFitArtifacts,
    pair_spec_id: str,
) -> DirectionalContrastPairSpec:
    matches = tuple(
        pair
        for pair in artifacts.spec.directional_pairs
        if pair.pair_spec_id == pair_spec_id
    )
    if len(matches) != 1:
        raise _contract_error(
            "Directional pair is not uniquely registered in the cross-fit "
            "specification",
            code="directional_target_program_pair_not_registered",
            field="pair_spec_id",
            remediation=(
                "Use one exact pair_spec_id from CrossFitSpec.directional_pairs"
            ),
        )
    return matches[0]


def _raw_views_by_receiver(
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
    universe: FrozenDirectionalTargetProgramUniverse,
) -> dict[str, _ReceiverRawRegistry]:
    folds = tuple(sorted(artifacts.folds, key=lambda item: item.fold_id))
    receiver_ids = artifacts.receiver_universe.receiver_ids
    values: dict[str, list[_FoldRawView]] = {receiver: [] for receiver in receiver_ids}
    supports: dict[str, list[ReceiverTrainingSupportRecord]] = {
        receiver: [] for receiver in receiver_ids
    }
    context_keys = folds[0].training.config.context_keys
    registered_context_weights = tuple(
        sorted(
            (
                node_context_fields(node, context_keys)[0],
                float(weight),
            )
            for node, weight in pair.forward_contrast.weights.items()
        )
    )
    for fold in folds:
        if fold.training.target_prior_content_id != universe.target_prior_content_id:
            raise _contract_error(
                "Target-program universe does not match the cross-fit TargetPrior",
                code="directional_target_program_prior_mismatch",
                field="target_prior_content_id",
                remediation="Freeze programs from the exact prior used by cross-fit",
            )
        responses = {
            response.artifact_id: response for response in fold.receiver_responses
        }
        applications = {
            application.training_response_id: application
            for application in fold.receiver_response_applications
        }
        bindings = tuple(
            binding
            for binding in fold.directional_response_bindings
            if binding.pair_spec_id == pair.pair_spec_id
        )
        support_by_receiver = {
            record.receiver_id: record for record in fold.receiver_training_support
        }
        if tuple(sorted(support_by_receiver)) != receiver_ids:
            raise _contract_error(
                "Receiver support registry does not cover the frozen run axis",
                code="directional_target_program_support_coverage_mismatch",
                field="receiver_training_support",
                remediation="Use intact cross-fit artifacts with complete support rows",
            )
        for receiver in receiver_ids:
            support = support_by_receiver[receiver]
            support._require_intact()
            supports[receiver].append(support)
        binding_by_receiver = {binding.receiver: binding for binding in bindings}
        expected_receivers = tuple(
            record.receiver_id
            for record in fold.receiver_training_support
            if record.status is ReceiverTrainingSupportStatus.OBSERVED
        )
        if len(binding_by_receiver) != len(bindings) or (
            tuple(sorted(binding_by_receiver)) != expected_receivers
        ):
            raise _contract_error(
                "Directional binding registry does not match observed receiver support",
                code="directional_target_program_binding_coverage_mismatch",
                field="directional_response_bindings",
                remediation="Rebuild the intact directional cross-fit registry",
            )
        for receiver in receiver_ids:
            support = support_by_receiver[receiver]
            if support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
                continue
            binding = binding_by_receiver[receiver]
            binding._require_intact()
            try:
                forward_response = responses[binding.forward_response_id]
                reverse_response = responses[binding.reverse_response_id]
                forward_application = applications[binding.forward_response_id]
                reverse_application = applications[binding.reverse_response_id]
            except KeyError as error:
                raise _contract_error(
                    "Directional binding is missing its raw held-out response parent",
                    code="directional_target_program_response_parent_missing",
                    field="receiver_response_applications",
                    remediation="Use the intact in-memory CrossFitArtifacts parent",
                ) from error
            forward_response._require_intact()
            reverse_response._require_intact()
            forward_application._require_intact()
            reverse_application._require_intact()
            shared_response_scope = (
                type(forward_response),
                forward_response.receiver,
                forward_response.fold_id,
                forward_response.feature_ids,
                forward_response.context_keys,
                forward_response.value_scale,
            )
            reverse_response_scope = (
                type(reverse_response),
                reverse_response.receiver,
                reverse_response.fold_id,
                reverse_response.feature_ids,
                reverse_response.context_keys,
                reverse_response.value_scale,
            )
            shared_application_scope = (
                type(forward_application),
                forward_application.receiver,
                forward_application.fold_id,
                forward_application.feature_ids,
                forward_application.sample_ids,
                forward_application.sample_subject_ids,
                forward_application.sample_context_ids,
                forward_application.subject_ids,
                forward_application.sample_values_digest,
                forward_application.value_scale,
            )
            reverse_application_scope = (
                type(reverse_application),
                reverse_application.receiver,
                reverse_application.fold_id,
                reverse_application.feature_ids,
                reverse_application.sample_ids,
                reverse_application.sample_subject_ids,
                reverse_application.sample_context_ids,
                reverse_application.subject_ids,
                reverse_application.sample_values_digest,
                reverse_application.value_scale,
            )
            if shared_response_scope != reverse_response_scope or (
                shared_application_scope != reverse_application_scope
            ):
                raise _contract_error(
                    "Forward and reverse raw response parents are not on one scale",
                    code="directional_target_program_common_scale_mismatch",
                    field="forward_response,reverse_response",
                    remediation="Use exact forward/reverse applications from one pair",
                )
            if forward_application.feature_ids != universe.feature_ids:
                raise _contract_error(
                    "Target programs do not align to the response feature order",
                    code="directional_target_program_feature_mismatch",
                    field="feature_ids",
                    remediation="Refreeze the universe on the exact response features",
                )
            if forward_application.value_scale != _VALUE_SCALE:
                raise _contract_error(
                    "Directional target-program scores require common log1p CPM values",
                    code="directional_target_program_value_scale_mismatch",
                    field="value_scale",
                    remediation="Use the held-out fold gene-response applications",
                )
            if not np.array_equal(
                forward_application.sample_values,
                reverse_application.sample_values,
                equal_nan=True,
            ):
                raise _contract_error(
                    "Forward and reverse held-out expression matrices differ",
                    code="directional_target_program_common_scale_mismatch",
                    field="sample_values_digest",
                    remediation="Reapply both contrasts to the same held-out aggregate",
                )
            context_weights = tuple(
                sorted(
                    (
                        node_context_fields(node, forward_response.context_keys)[0],
                        float(weight),
                    )
                    for node, weight in pair.forward_contrast.weights.items()
                )
            )
            if context_weights != registered_context_weights:
                raise _contract_error(
                    "Directional response context encoding differs from the run config",
                    code="directional_target_program_context_lineage_mismatch",
                    field="context_weights",
                    remediation="Use responses produced from the intact root config",
                )
            values[receiver].append(
                _FoldRawView(
                    fold_id=fold.fold_id,
                    forward_application_id=forward_application.application_id,
                    reverse_application_id=reverse_application.application_id,
                    context_weights=context_weights,
                    sample_ids=forward_application.sample_ids,
                    sample_subject_ids=forward_application.sample_subject_ids,
                    sample_context_ids=forward_application.sample_context_ids,
                    values=forward_application.sample_values,
                )
            )
    return {
        receiver: _ReceiverRawRegistry(
            support_records=tuple(supports[receiver]),
            views=tuple(sorted(values[receiver], key=lambda item: item.fold_id)),
            context_weights=registered_context_weights,
        )
        for receiver in receiver_ids
    }


def _source_digests(
    selected_rows: list[tuple[str, str, str, str]],
    selected_values: np.ndarray,
) -> tuple[str, str]:
    row_digest = stable_id(
        "directional_target_program_oof_rows",
        [list(row) for row in selected_rows],
        schema_version=_SCHEMA_VERSION,
        digest_length=64,
    )
    return row_digest, _array_digest(selected_values)


def _not_estimable_effect(
    *,
    receiver: str,
    feature_ids: tuple[str, ...],
    pair_spec_id: str,
    score_spec_id: str,
    receiver_universe_id: str,
    receiver_axis_id: str,
    registry: _ReceiverRawRegistry,
    selected_rows: list[tuple[str, str, str, str]],
    selected_values: np.ndarray,
    n_subject_context_rows: int,
    n_subjects: int,
    design_rank: int,
    reason_code: str,
    design_condition_number: float | None = None,
) -> DirectionalTargetProgramEffect:
    views = registry.views
    row_digest, values_digest = _source_digests(selected_rows, selected_values)
    return DirectionalTargetProgramEffect._from_fit(
        receiver=receiver,
        feature_ids=feature_ids,
        pair_spec_id=pair_spec_id,
        score_spec_id=score_spec_id,
        receiver_universe_id=receiver_universe_id,
        receiver_axis_id=receiver_axis_id,
        fold_ids=tuple(record.outer_fold_id for record in registry.support_records),
        receiver_training_support=registry.support_records,
        response_application_fold_ids=tuple(view.fold_id for view in views),
        forward_response_application_ids=tuple(
            view.forward_application_id for view in views
        ),
        reverse_response_application_ids=tuple(
            view.reverse_application_id for view in views
        ),
        context_weights=registry.context_weights,
        source_row_manifest_digest=row_digest,
        source_values_digest=values_digest,
        gene_effect=np.full(len(feature_ids), np.nan, dtype=float),
        n_samples=len(selected_rows),
        n_subject_context_rows=n_subject_context_rows,
        n_subjects=n_subjects,
        design_rank=design_rank,
        design_condition_number=design_condition_number,
        status=_STATUS_NOT_ESTIMABLE,
        reason_code=reason_code,
    )


def _fit_receiver_effect(
    receiver: str,
    feature_ids: tuple[str, ...],
    pair_spec_id: str,
    receiver_universe_id: str,
    receiver_axis_id: str,
    registry: _ReceiverRawRegistry,
    spec: DirectionalTargetProgramScoreSpec,
) -> DirectionalTargetProgramEffect:
    views = registry.views
    if any(
        record.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE
        for record in registry.support_records
    ):
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=[],
            selected_values=np.empty((0, len(feature_ids)), dtype=float),
            n_subject_context_rows=0,
            n_subjects=0,
            design_rank=0,
            reason_code=_RECEIVER_ABSENT_REASON,
        )
    if not views:
        raise RuntimeError("supported receiver effect requires fold views")
    context_sets = {view.context_weights for view in views}
    if len(context_sets) != 1:
        raise _contract_error(
            "Directional contrast context weights differ across outer folds",
            code="directional_target_program_fold_contrast_mismatch",
            field="context_weights",
            remediation="Use one intact directional pair across every outer fold",
        )
    context_weights = next(iter(context_sets))
    if context_weights != registry.context_weights:
        raise _contract_error(
            "Directional context weights differ from registered receiver lineage",
            code="directional_target_program_fold_contrast_mismatch",
            field="context_weights",
            remediation="Use one intact directional pair across every outer fold",
        )
    context_ids = tuple(context for context, _ in context_weights)
    context_universe = set(context_ids)
    selected_rows: list[tuple[str, str, str, str]] = []
    selected_vectors: list[np.ndarray] = []
    for view in views:
        for index, (sample_id, subject_id, context_id) in enumerate(
            zip(
                view.sample_ids,
                view.sample_subject_ids,
                view.sample_context_ids,
                strict=True,
            )
        ):
            if context_id not in context_universe:
                continue
            selected_rows.append((view.fold_id, sample_id, subject_id, context_id))
            selected_vectors.append(np.asarray(view.values[index], dtype=float))
    selected_values = (
        np.vstack(selected_vectors)
        if selected_vectors
        else np.empty((0, len(feature_ids)), dtype=float)
    )
    if len({row[1] for row in selected_rows}) != len(selected_rows):
        raise _contract_error(
            "A held-out sample appears in more than one directional outer fold",
            code="directional_target_program_oof_sample_leakage",
            field="sample_id",
            remediation="Repair the subject-blocked outer fold plan",
        )
    folds_by_subject: dict[str, set[str]] = defaultdict(set)
    for fold_id, _, subject_id, _ in selected_rows:
        folds_by_subject[subject_id].add(fold_id)
    if any(len(fold_ids) != 1 for fold_ids in folds_by_subject.values()):
        raise _contract_error(
            "A subject appears in more than one directional outer fold",
            code="directional_target_program_oof_subject_leakage",
            field="subject_id,fold_id",
            remediation="Repair the subject-blocked outer fold plan",
        )
    n_subjects = len(folds_by_subject)
    if not selected_rows or np.any(~np.isfinite(selected_values)):
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=0,
            n_subjects=n_subjects,
            design_rank=0,
            reason_code="directional_target_program_incomplete_oof_response_coverage",
        )
    observed_contexts = {row[3] for row in selected_rows}
    if observed_contexts != context_universe:
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=0,
            n_subjects=n_subjects,
            design_rank=0,
            reason_code="directional_target_program_context_universe_mismatch",
        )
    contexts_by_fold: dict[str, set[str]] = defaultdict(set)
    for fold_id, _, _, context_id in selected_rows:
        contexts_by_fold[fold_id].add(context_id)
    registered_folds = {view.fold_id for view in views}
    if set(contexts_by_fold) != registered_folds or any(
        contexts != context_universe for contexts in contexts_by_fold.values()
    ):
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=0,
            n_subjects=n_subjects,
            design_rank=0,
            reason_code="directional_target_program_fold_context_support_incomplete",
        )

    grouped_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (fold_id, _, subject_id, context_id) in enumerate(selected_rows):
        grouped_indices[(subject_id, fold_id, context_id)].append(index)
    collapsed_keys = tuple(sorted(grouped_indices))
    collapsed_values = np.vstack(
        [selected_values[grouped_indices[key]].mean(axis=0) for key in collapsed_keys]
    )
    n_subject_context_rows = len(collapsed_keys)
    if n_subjects < 2:
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=n_subject_context_rows,
            n_subjects=n_subjects,
            design_rank=0,
            reason_code="directional_target_program_insufficient_subject_clusters",
        )
    folds = tuple(sorted({key[1] for key in collapsed_keys}))
    context_index = {context: index for index, context in enumerate(context_ids)}
    fold_index = {
        fold_id: len(context_ids) + index for index, fold_id in enumerate(folds[1:])
    }
    design: np.ndarray = np.zeros(
        (len(collapsed_keys), len(context_ids) + max(0, len(folds) - 1)),
        dtype=float,
    )
    for row_index, (_, fold_id, context_id) in enumerate(collapsed_keys):
        design[row_index, context_index[context_id]] = 1.0
        if fold_id in fold_index:
            design[row_index, fold_index[fold_id]] = 1.0
    subject_counts = Counter(key[0] for key in collapsed_keys)
    row_weights = np.asarray(
        [1.0 / subject_counts[key[0]] for key in collapsed_keys],
        dtype=float,
    )
    weighted_design = np.sqrt(row_weights)[:, np.newaxis] * design
    weighted_values = np.sqrt(row_weights)[:, np.newaxis] * collapsed_values
    rank = int(np.linalg.matrix_rank(weighted_design))
    if rank != design.shape[1]:
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=n_subject_context_rows,
            n_subjects=n_subjects,
            design_rank=rank,
            reason_code="directional_target_program_rank_deficient_design",
        )
    condition_number = float(np.linalg.cond(weighted_design))
    if (
        not math.isfinite(condition_number)
        or condition_number > spec.maximum_condition_number
    ):
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=n_subject_context_rows,
            n_subjects=n_subjects,
            design_rank=rank,
            reason_code="directional_target_program_ill_conditioned_design",
            design_condition_number=condition_number,
        )
    gram = weighted_design.T @ weighted_design
    try:
        coefficients = np.linalg.solve(gram, weighted_design.T @ weighted_values)
    except np.linalg.LinAlgError:
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=n_subject_context_rows,
            n_subjects=n_subjects,
            design_rank=rank,
            reason_code="directional_target_program_singular_weighted_gram",
            design_condition_number=condition_number,
        )
    contrast: np.ndarray = np.zeros(design.shape[1], dtype=float)
    for context_id, weight in context_weights:
        contrast[context_index[context_id]] = weight
    gene_effect = np.asarray(contrast @ coefficients, dtype=float)
    if gene_effect.shape != (len(feature_ids),) or np.any(~np.isfinite(gene_effect)):
        return _not_estimable_effect(
            receiver=receiver,
            feature_ids=feature_ids,
            pair_spec_id=pair_spec_id,
            score_spec_id=spec.spec_id,
            receiver_universe_id=receiver_universe_id,
            receiver_axis_id=receiver_axis_id,
            registry=registry,
            selected_rows=selected_rows,
            selected_values=selected_values,
            n_subject_context_rows=n_subject_context_rows,
            n_subjects=n_subjects,
            design_rank=rank,
            reason_code="directional_target_program_nonfinite_gene_effect",
            design_condition_number=condition_number,
        )
    row_digest, values_digest = _source_digests(selected_rows, selected_values)
    return DirectionalTargetProgramEffect._from_fit(
        receiver=receiver,
        feature_ids=feature_ids,
        pair_spec_id=pair_spec_id,
        score_spec_id=spec.spec_id,
        receiver_universe_id=receiver_universe_id,
        receiver_axis_id=receiver_axis_id,
        fold_ids=tuple(record.outer_fold_id for record in registry.support_records),
        receiver_training_support=registry.support_records,
        response_application_fold_ids=tuple(view.fold_id for view in views),
        forward_response_application_ids=tuple(
            view.forward_application_id for view in views
        ),
        reverse_response_application_ids=tuple(
            view.reverse_application_id for view in views
        ),
        context_weights=context_weights,
        source_row_manifest_digest=row_digest,
        source_values_digest=values_digest,
        gene_effect=gene_effect,
        n_samples=len(selected_rows),
        n_subject_context_rows=n_subject_context_rows,
        n_subjects=n_subjects,
        design_rank=rank,
        design_condition_number=condition_number,
        status=_STATUS_OBSERVED,
        reason_code=None,
    )


def _score_table(
    *,
    crossfit_id: str,
    pair: DirectionalContrastPairSpec,
    universe: FrozenDirectionalTargetProgramUniverse,
    spec: DirectionalTargetProgramScoreSpec,
    effects: tuple[DirectionalTargetProgramEffect, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    profiles = universe._normalized_profiles
    channels = (
        pair.forward_channel_name,
        pair.reverse_channel_name,
    )
    for effect in effects:
        effect._require_intact()
        channel_vectors: tuple[np.ndarray | None, np.ndarray | None]
        if effect.status == _STATUS_OBSERVED:
            channel_vectors = (
                np.maximum(effect.gene_effect, 0.0),
                np.maximum(-effect.gene_effect, 0.0),
            )
        else:
            channel_vectors = (None, None)
        for channel, channel_vector in zip(channels, channel_vectors, strict=True):
            if channel_vector is None:
                scores: np.ndarray | None = None
            else:
                channel_norm = float(np.linalg.norm(channel_vector))
                if channel_norm == 0.0:
                    scores = np.zeros(len(universe.program_ids), dtype=float)
                else:
                    scores = np.asarray(
                        profiles.T @ (channel_vector / channel_norm),
                        dtype=float,
                    ).reshape(-1)
                    scores = np.clip(scores, 0.0, 1.0)
            for index, program_id in enumerate(universe.program_ids):
                matched_count = universe.matched_target_counts[index]
                if effect.status != _STATUS_OBSERVED:
                    status = _STATUS_NOT_ESTIMABLE
                    reason = effect.reason_code
                    score = np.nan
                elif matched_count < spec.minimum_matched_targets:
                    status = _STATUS_NOT_ESTIMABLE
                    reason = "directional_target_program_insufficient_matched_targets"
                    score = np.nan
                else:
                    if scores is None:  # pragma: no cover - status invariant
                        raise RuntimeError("observed effect is missing channel scores")
                    status = _STATUS_OBSERVED
                    reason = None
                    score = float(scores[index])
                rows.append(
                    {
                        "crossfit_id": crossfit_id,
                        "receiver_universe_id": effect.receiver_universe_id,
                        "receiver_axis_id": effect.receiver_axis_id,
                        "receiver_training_support_ids": (
                            effect.receiver_training_support_ids
                        ),
                        "pair_spec_id": pair.pair_spec_id,
                        "target_program_universe_id": universe.universe_id,
                        "score_spec_id": spec.spec_id,
                        "analysis_track": _ANALYSIS_TRACK,
                        "sender": _SOURCE_AGNOSTIC_SENDER,
                        "receiver": effect.receiver,
                        "program_id": program_id,
                        "channel": channel,
                        "score": score,
                        "score_direction": "higher",
                        "status": status,
                        "reason_code": reason,
                        "matched_target_count": matched_count,
                        "effect_result_id": effect.result_id,
                        "effect_backend": spec.effect_backend,
                        "score_method": spec.score_method,
                        "value_scale": spec.value_scale,
                        "supports_active_inhibition_claim": False,
                        "formal_inference_allowed": False,
                    }
                )
    return pd.DataFrame(rows, columns=_SCORE_COLUMNS).sort_values(
        ["receiver", "program_id", "channel"],
        kind="stable",
        ignore_index=True,
    )


@dataclass(frozen=True, slots=True, init=False)
class DirectionalTargetProgramScoreCollection:
    """Complete receiver x program x channel common-scale score collection."""

    collection_id: str
    crossfit_id: str
    crossfit_spec_id: str
    receiver_universe_id: str
    receiver_axis_id: str
    pair_spec_id: str
    target_program_universe_id: str
    score_spec_id: str
    receiver_ids: tuple[str, ...]
    program_ids: tuple[str, ...]
    channels: tuple[str, ...]
    effect_result_ids: tuple[str, ...]
    analysis_track: str
    sender_scope: str
    score_method: str
    effect_backend: str
    value_scale: str
    supports_active_inhibition_claim: bool
    formal_inference_allowed: bool
    _effects: tuple[DirectionalTargetProgramEffect, ...]
    _pair_spec: DirectionalContrastPairSpec
    _universe: FrozenDirectionalTargetProgramUniverse
    _score_spec: DirectionalTargetProgramScoreSpec
    _source_artifacts: CrossFitArtifacts
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "DirectionalTargetProgramScoreCollection is producer-owned; use "
            "score_crossfit_directional_target_programs()"
        )

    @classmethod
    def _from_effects(
        cls,
        artifacts: CrossFitArtifacts,
        pair: DirectionalContrastPairSpec,
        universe: FrozenDirectionalTargetProgramUniverse,
        score_spec: DirectionalTargetProgramScoreSpec,
        effects: tuple[DirectionalTargetProgramEffect, ...],
    ) -> DirectionalTargetProgramScoreCollection:
        receiver_ids = tuple(effect.receiver for effect in effects)
        effect_ids = tuple(effect.result_id for effect in effects)
        self = object.__new__(cls)
        values: dict[str, object] = {
            "crossfit_id": artifacts.crossfit_id,
            "crossfit_spec_id": artifacts.spec.spec_id,
            "receiver_universe_id": artifacts.receiver_universe.universe_id,
            "receiver_axis_id": artifacts.receiver_universe.receiver_axis_id,
            "pair_spec_id": pair.pair_spec_id,
            "target_program_universe_id": universe.universe_id,
            "score_spec_id": score_spec.spec_id,
            "receiver_ids": receiver_ids,
            "program_ids": universe.program_ids,
            "channels": (
                pair.forward_channel_name,
                pair.reverse_channel_name,
            ),
            "effect_result_ids": effect_ids,
            "analysis_track": _ANALYSIS_TRACK,
            "sender_scope": _SOURCE_AGNOSTIC_SENDER,
            "score_method": score_spec.score_method,
            "effect_backend": score_spec.effect_backend,
            "value_scale": score_spec.value_scale,
            "supports_active_inhibition_claim": False,
            "formal_inference_allowed": False,
            "_effects": effects,
            "_pair_spec": pair,
            "_universe": universe,
            "_score_spec": score_spec,
            "_source_artifacts": artifacts,
            "_producer_marker": _COLLECTION_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "directional_target_program_score_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "analysis_track": self.analysis_track,
            "channels": list(self.channels),
            "crossfit_id": self.crossfit_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "effect_backend": self.effect_backend,
            "effect_result_ids": list(self.effect_result_ids),
            "formal_inference_allowed": self.formal_inference_allowed,
            "pair_spec_id": self.pair_spec_id,
            "program_ids": list(self.program_ids),
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_ids": list(self.receiver_ids),
            "receiver_universe_id": self.receiver_universe_id,
            "score_method": self.score_method,
            "score_spec_id": self.score_spec_id,
            "sender_scope": self.sender_scope,
            "supports_active_inhibition_claim": (self.supports_active_inhibition_claim),
            "target_program_universe_id": self.target_program_universe_id,
            "value_scale": self.value_scale,
        }

    def _require_intact(self) -> None:
        try:
            self._source_artifacts._require_intact()
            self._pair_spec._require_intact()
            self._universe._require_intact()
            self._score_spec._require_intact()
            effects = tuple(self._effects)
            for effect in effects:
                effect._require_intact()
            receiver_ids = tuple(effect.receiver for effect in effects)
            raw_registry = _raw_views_by_receiver(
                self._source_artifacts,
                self._pair_spec,
                self._universe,
            )
            parent_lineage_valid = all(
                effect.receiver in raw_registry
                and effect.fold_ids
                == tuple(
                    record.outer_fold_id
                    for record in raw_registry[effect.receiver].support_records
                )
                and effect.receiver_training_support_ids
                == tuple(
                    record.support_record_id
                    for record in raw_registry[effect.receiver].support_records
                )
                and effect.response_application_fold_ids
                == tuple(view.fold_id for view in raw_registry[effect.receiver].views)
                and effect.forward_response_application_ids
                == tuple(
                    view.forward_application_id
                    for view in raw_registry[effect.receiver].views
                )
                and effect.reverse_response_application_ids
                == tuple(
                    view.reverse_application_id
                    for view in raw_registry[effect.receiver].views
                )
                and effect.context_weights
                == raw_registry[effect.receiver].context_weights
                for effect in effects
            )
            expected_id = stable_id(
                "directional_target_program_score_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.crossfit_id == self._source_artifacts.crossfit_id
                and self.crossfit_spec_id == self._source_artifacts.spec.spec_id
                and self.receiver_universe_id
                == self._source_artifacts.receiver_universe.universe_id
                and self.receiver_axis_id
                == self._source_artifacts.receiver_universe.receiver_axis_id
                and self.pair_spec_id == self._pair_spec.pair_spec_id
                and self.target_program_universe_id == self._universe.universe_id
                and self.score_spec_id == self._score_spec.spec_id
                and self.receiver_ids == receiver_ids
                and self.receiver_ids
                == self._source_artifacts.receiver_universe.receiver_ids
                and self.program_ids == self._universe.program_ids
                and self.channels
                == (
                    self._pair_spec.forward_channel_name,
                    self._pair_spec.reverse_channel_name,
                )
                and self.channels == _CHANNELS
                and self.effect_result_ids
                == tuple(effect.result_id for effect in effects)
                and parent_lineage_valid
                and all(
                    effect.feature_ids == self._universe.feature_ids
                    and effect.pair_spec_id == self.pair_spec_id
                    and effect.score_spec_id == self.score_spec_id
                    and effect.receiver_universe_id == self.receiver_universe_id
                    and effect.receiver_axis_id == self.receiver_axis_id
                    for effect in effects
                )
                and self.analysis_track == _ANALYSIS_TRACK
                and self.sender_scope == _SOURCE_AGNOSTIC_SENDER
                and self.score_method == _SCORE_METHOD
                and self.effect_backend == _EFFECT_BACKEND
                and self.value_scale == _VALUE_SCALE
                and self.supports_active_inhibition_claim is False
                and self.formal_inference_allowed is False
                and expected_id == self.collection_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _contract_error(
                "Directional target-program collection failed integrity validation",
                code="directional_target_program_collection_integrity_violation",
                field="collection_id",
                remediation=(
                    "Recompute scores from intact in-memory cross-fit artifacts"
                ),
            ) from error
        if not valid:
            raise _contract_error(
                "Directional target-program collection failed integrity validation",
                code="directional_target_program_collection_integrity_violation",
                field="collection_id",
                remediation=(
                    "Recompute scores from intact in-memory cross-fit artifacts"
                ),
            )

    @property
    def effects(self) -> tuple[DirectionalTargetProgramEffect, ...]:
        self._require_intact()
        return self._effects

    @property
    def gene_effects(self) -> pd.DataFrame:
        """Return feature x receiver signed effects; unavailable columns are NaN."""

        self._require_intact()
        return pd.DataFrame(
            {effect.receiver: effect.gene_effect for effect in self._effects},
            index=pd.Index(self._universe.feature_ids, name="feature_id"),
        )

    @property
    def scores(self) -> pd.DataFrame:
        """Return the complete source-agnostic two-channel score table."""

        self._require_intact()
        return _score_table(
            crossfit_id=self.crossfit_id,
            pair=self._pair_spec,
            universe=self._universe,
            spec=self._score_spec,
            effects=self._effects,
        )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        scores = self.scores
        return {
            "collection_id": self.collection_id,
            **self._identity_payload(),
            "score_status_counts": {
                str(status): int(count)
                for status, count in scores["status"].value_counts().items()
            },
            "score_spec": self._score_spec.to_dict(),
            "target_program_universe": self._universe.to_dict(),
            "effects": [effect.to_dict() for effect in self._effects],
            "excluded_output_kinds": [
                "p_value",
                "q_value",
                "posterior_probability",
                "communication_probability",
            ],
        }


def score_crossfit_directional_target_programs(
    artifacts: CrossFitArtifacts,
    universe: FrozenDirectionalTargetProgramUniverse,
    *,
    pair_spec_id: str,
    score_spec: DirectionalTargetProgramScoreSpec | None = None,
) -> DirectionalTargetProgramScoreCollection:
    """Produce complete common-scale OOF scores for one registered pair."""

    if not isinstance(artifacts, CrossFitArtifacts):
        raise TypeError("artifacts must be CrossFitArtifacts")
    if not isinstance(universe, FrozenDirectionalTargetProgramUniverse):
        raise TypeError("universe must be FrozenDirectionalTargetProgramUniverse")
    _name(pair_spec_id, field_name="pair_spec_id")
    spec = score_spec or DirectionalTargetProgramScoreSpec()
    if not isinstance(spec, DirectionalTargetProgramScoreSpec):
        raise TypeError("score_spec must be DirectionalTargetProgramScoreSpec or None")
    artifacts._require_intact()
    universe._require_intact()
    spec._require_intact()
    pair = _directional_pair(artifacts, pair_spec_id)
    pair._require_intact()
    views_by_receiver = _raw_views_by_receiver(artifacts, pair, universe)
    if not views_by_receiver:
        raise _contract_error(
            "Directional pair has no receiver applications",
            code="directional_target_program_receiver_registry_empty",
            field="directional_response_bindings",
            remediation="Run cross-fit with the explicit directional pair",
        )
    effects = tuple(
        _fit_receiver_effect(
            receiver,
            universe.feature_ids,
            pair.pair_spec_id,
            artifacts.receiver_universe.universe_id,
            artifacts.receiver_universe.receiver_axis_id,
            registry,
            spec,
        )
        for receiver, registry in views_by_receiver.items()
    )
    return DirectionalTargetProgramScoreCollection._from_effects(
        artifacts,
        pair,
        universe,
        spec,
        effects,
    )


__all__ = [
    "DirectionalTargetProgramEffect",
    "DirectionalTargetProgramScoreCollection",
    "DirectionalTargetProgramScoreSpec",
    "FrozenDirectionalTargetProgramUniverse",
    "freeze_crossfit_directional_target_program_universe",
    "freeze_directional_target_program_universe",
    "score_crossfit_directional_target_programs",
]
