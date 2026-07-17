"""Run-level receiver axis and fold-training support contracts.

The receiver axis is frozen before any context-specific fitting.  Its default
source is the context-label-blind set of cell-type identifiers observed in the
sanitized root input.  A caller may instead predeclare a superset, but observed
root cell types can never fall outside that declaration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from crychic.core import ContractError, stable_id

from .training import SanitizedRawInputIdentity

_SCHEMA_VERSION: Final = "1"
_UNIVERSE_PRODUCER: Final = "crychic.workflow.frozen_receiver_universe.v1"
_SUPPORT_PRODUCER: Final = "crychic.workflow.receiver_training_support.v1"
_REUSE_PRODUCER: Final = "crychic.workflow.receiver_universe_reuse_binding.v1"
_ABSENT_REASON: Final = "receiver_absent_in_outer_training"


class ReceiverUniverseSourcePolicy(StrEnum):
    """Pre-fit policy that owns the immutable receiver axis."""

    ROOT_OBSERVED_CONTEXT_LABEL_BLIND_V1 = (
        "root_observed_cell_type_ids_context_label_blind_v1"
    )
    EXPLICIT_PREDECLARED_V1 = "explicit_predeclared_receiver_ids_v1"


class ReceiverTrainingSupportStatus(StrEnum):
    """Whether one frozen receiver is supported in an outer-training fold."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _canonical_ids(
    values: Sequence[str],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    raw = tuple(_name(value, field_name=field_name) for value in values)
    if not raw and not allow_empty:
        raise ValueError(f"{field_name} must be non-empty")
    if len(raw) != len(set(raw)):
        raise ValueError(f"{field_name} must contain unique identifiers")
    return tuple(sorted(raw))


def _integrity_error(message: str, *, field: str) -> ContractError:
    return ContractError(
        message,
        code="frozen_receiver_universe_integrity_violation",
        field=field,
        remediation=(
            "Refreeze the receiver universe from the intact sanitized root input"
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenReceiverUniverse:
    """Producer-owned run-level receiver universe with exact root lineage."""

    receiver_ids: tuple[str, ...]
    receiver_axis_id: str
    observed_cell_type_ids: tuple[str, ...]
    source_policy: ReceiverUniverseSourcePolicy
    root_input_identity_id: str
    root_input_digest: str
    root_config_digest: str
    root_subject_ids: tuple[str, ...]
    universe_id: str
    _root_input_identity: SanitizedRawInputIdentity = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "FrozenReceiverUniverse is producer-owned; use "
            "freeze_receiver_universe()"
        )

    @classmethod
    def _from_root(
        cls,
        root_input_identity: SanitizedRawInputIdentity,
        *,
        observed_cell_type_ids: tuple[str, ...],
        receiver_ids: tuple[str, ...],
        source_policy: ReceiverUniverseSourcePolicy,
    ) -> FrozenReceiverUniverse:
        root_input_identity._require_intact()
        self = object.__new__(cls)
        object.__setattr__(self, "receiver_ids", receiver_ids)
        object.__setattr__(
            self,
            "receiver_axis_id",
            stable_id(
                "receiver_axis",
                {"receiver_ids": list(receiver_ids)},
                schema_version=_SCHEMA_VERSION,
            ),
        )
        object.__setattr__(self, "observed_cell_type_ids", observed_cell_type_ids)
        object.__setattr__(self, "source_policy", source_policy)
        object.__setattr__(
            self,
            "root_input_identity_id",
            root_input_identity.identity_id,
        )
        object.__setattr__(self, "root_input_digest", root_input_identity.input_digest)
        object.__setattr__(
            self,
            "root_config_digest",
            root_input_identity.config_digest,
        )
        object.__setattr__(
            self,
            "root_subject_ids",
            tuple(root_input_identity.subject_ids),
        )
        object.__setattr__(self, "_root_input_identity", root_input_identity)
        object.__setattr__(self, "_producer_marker", _UNIVERSE_PRODUCER)
        object.__setattr__(
            self,
            "universe_id",
            stable_id(
                "frozen_receiver_universe",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "observed_cell_type_ids": list(self.observed_cell_type_ids),
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_ids": list(self.receiver_ids),
            "root_config_digest": self.root_config_digest,
            "root_input_digest": self.root_input_digest,
            "root_input_identity_id": self.root_input_identity_id,
            "root_subject_ids": list(self.root_subject_ids),
            "source_policy": self.source_policy.value,
        }

    def _require_intact(self) -> None:
        try:
            if not isinstance(self._root_input_identity, SanitizedRawInputIdentity):
                raise TypeError("invalid root identity type")
            self._root_input_identity._require_intact()
            receivers = _canonical_ids(
                self.receiver_ids,
                field_name="receiver_ids",
            )
            observed = _canonical_ids(
                self.observed_cell_type_ids,
                field_name="observed_cell_type_ids",
            )
            if not isinstance(self.source_policy, ReceiverUniverseSourcePolicy):
                raise TypeError("invalid receiver universe source policy")
            if self.source_policy is (
                ReceiverUniverseSourcePolicy.ROOT_OBSERVED_CONTEXT_LABEL_BLIND_V1
            ):
                policy_valid = receivers == observed
            else:
                policy_valid = set(observed).issubset(receivers)
            expected_id = stable_id(
                "frozen_receiver_universe",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            expected_axis_id = stable_id(
                "receiver_axis",
                {"receiver_ids": list(receivers)},
                schema_version=_SCHEMA_VERSION,
            )
            root = self._root_input_identity
            valid = (
                self._producer_marker == _UNIVERSE_PRODUCER
                and receivers == self.receiver_ids
                and self.receiver_axis_id == expected_axis_id
                and observed == self.observed_cell_type_ids
                and policy_valid
                and self.root_input_identity_id == root.identity_id
                and self.root_input_digest == root.input_digest
                and self.root_config_digest == root.config_digest
                and self.root_subject_ids == root.subject_ids
                and self.universe_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise _integrity_error(
                "Frozen receiver universe failed integrity validation",
                field="universe_id",
            ) from error
        if not valid:
            raise _integrity_error(
                "Frozen receiver universe failed integrity validation",
                field="universe_id",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical universe contract after integrity validation."""

        self._require_intact()
        return {
            "universe_id": self.universe_id,
            **self._identity_payload(),
        }


@dataclass(frozen=True, slots=True, init=False)
class ReceiverUniverseReuseBinding:
    """Producer-owned proof that a resample reused an exact receiver axis."""

    source_receiver_universe_id: str
    child_receiver_universe_id: str
    receiver_axis_id: str
    receiver_ids: tuple[str, ...]
    source_root_input_identity_id: str
    child_root_input_identity_id: str
    child_root_input_digest: str
    child_root_config_digest: str
    operation: str
    plan_id: str
    resample_index: int
    materialized_input_id: str
    binding_id: str
    _source_universe: FrozenReceiverUniverse = field(repr=False)
    _child_universe: FrozenReceiverUniverse = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverUniverseReuseBinding is producer-owned; use "
            "bind_reused_receiver_universe()"
        )

    @classmethod
    def _from_resample(
        cls,
        source_universe: FrozenReceiverUniverse,
        child_universe: FrozenReceiverUniverse,
        *,
        operation: str,
        plan_id: str,
        resample_index: int,
        materialized_input_id: str,
    ) -> ReceiverUniverseReuseBinding:
        self = object.__new__(cls)
        object.__setattr__(
            self,
            "source_receiver_universe_id",
            source_universe.universe_id,
        )
        object.__setattr__(
            self,
            "child_receiver_universe_id",
            child_universe.universe_id,
        )
        object.__setattr__(self, "receiver_axis_id", source_universe.receiver_axis_id)
        object.__setattr__(self, "receiver_ids", source_universe.receiver_ids)
        object.__setattr__(
            self,
            "source_root_input_identity_id",
            source_universe.root_input_identity_id,
        )
        object.__setattr__(
            self,
            "child_root_input_identity_id",
            child_universe.root_input_identity_id,
        )
        object.__setattr__(
            self,
            "child_root_input_digest",
            child_universe.root_input_digest,
        )
        object.__setattr__(
            self,
            "child_root_config_digest",
            child_universe.root_config_digest,
        )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "resample_index", resample_index)
        object.__setattr__(self, "materialized_input_id", materialized_input_id)
        object.__setattr__(self, "_source_universe", source_universe)
        object.__setattr__(self, "_child_universe", child_universe)
        object.__setattr__(self, "_producer_marker", _REUSE_PRODUCER)
        object.__setattr__(
            self,
            "binding_id",
            stable_id(
                "receiver_universe_reuse_binding",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "source_receiver_universe_id": self.source_receiver_universe_id,
            "child_receiver_universe_id": self.child_receiver_universe_id,
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_ids": list(self.receiver_ids),
            "source_root_input_identity_id": self.source_root_input_identity_id,
            "child_root_input_identity_id": self.child_root_input_identity_id,
            "child_root_input_digest": self.child_root_input_digest,
            "child_root_config_digest": self.child_root_config_digest,
            "operation": self.operation,
            "plan_id": self.plan_id,
            "resample_index": self.resample_index,
            "materialized_input_id": self.materialized_input_id,
        }

    def _require_intact(self) -> None:
        try:
            if not isinstance(self._source_universe, FrozenReceiverUniverse) or not (
                isinstance(self._child_universe, FrozenReceiverUniverse)
            ):
                raise TypeError("invalid receiver universe type")
            self._source_universe._require_intact()
            self._child_universe._require_intact()
            operation = _name(self.operation, field_name="operation")
            plan_id = _name(self.plan_id, field_name="plan_id")
            input_id = _name(
                self.materialized_input_id,
                field_name="materialized_input_id",
            )
            if (
                isinstance(self.resample_index, bool)
                or not isinstance(self.resample_index, int)
                or self.resample_index < 0
            ):
                raise ValueError("resample_index must be a non-negative integer")
            receivers = _canonical_ids(self.receiver_ids, field_name="receiver_ids")
            source = self._source_universe
            child = self._child_universe
            expected_id = stable_id(
                "receiver_universe_reuse_binding",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _REUSE_PRODUCER
                and operation == self.operation
                and plan_id == self.plan_id
                and input_id == self.materialized_input_id
                and receivers == self.receiver_ids == source.receiver_ids
                and child.receiver_ids == source.receiver_ids
                and child.receiver_axis_id == source.receiver_axis_id
                and self.receiver_axis_id == source.receiver_axis_id
                and self.source_receiver_universe_id == source.universe_id
                and self.child_receiver_universe_id == child.universe_id
                and self.source_root_input_identity_id
                == source.root_input_identity_id
                and self.child_root_input_identity_id == child.root_input_identity_id
                and self.child_root_input_digest == child.root_input_digest
                and self.child_root_config_digest == child.root_config_digest
                and self.binding_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-universe reuse binding failed integrity validation",
                code="receiver_universe_reuse_binding_integrity_violation",
                field="binding_id",
                remediation=(
                    "Rebind the intact source and child receiver universes from "
                    "the exact materialized resampling plan"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Receiver-universe reuse binding failed integrity validation",
                code="receiver_universe_reuse_binding_integrity_violation",
                field="binding_id",
                remediation=(
                    "Rebind the intact source and child receiver universes from "
                    "the exact materialized resampling plan"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical resampling-axis lineage contract."""

        self._require_intact()
        return {"binding_id": self.binding_id, **self._identity_payload()}


def bind_reused_receiver_universe(
    source_universe: FrozenReceiverUniverse,
    child_universe: FrozenReceiverUniverse,
    *,
    operation: str,
    plan_id: str,
    resample_index: int,
    materialized_input_id: str,
) -> ReceiverUniverseReuseBinding:
    """Bind a root-specific child universe to an unchanged source receiver axis."""

    if not isinstance(source_universe, FrozenReceiverUniverse) or not isinstance(
        child_universe, FrozenReceiverUniverse
    ):
        raise TypeError("source_universe and child_universe must be frozen universes")
    source_universe._require_intact()
    child_universe._require_intact()
    if (
        source_universe.receiver_axis_id != child_universe.receiver_axis_id
        or source_universe.receiver_ids != child_universe.receiver_ids
    ):
        raise ContractError(
            "Resampled child receiver axis differs from the frozen source axis",
            code="receiver_universe_reuse_axis_mismatch",
            field="receiver_axis_id",
            remediation=(
                "Pass the complete source receiver IDs into the child cross-fit "
                "before any resampling-specific model fitting"
            ),
        )
    return ReceiverUniverseReuseBinding._from_resample(
        source_universe,
        child_universe,
        operation=_name(operation, field_name="operation"),
        plan_id=_name(plan_id, field_name="plan_id"),
        resample_index=resample_index,
        materialized_input_id=_name(
            materialized_input_id,
            field_name="materialized_input_id",
        ),
    )


def freeze_receiver_universe(
    root_input_identity: SanitizedRawInputIdentity,
    observed_cell_type_ids: Sequence[str],
    *,
    predeclared_receiver_ids: Sequence[str] | None = None,
) -> FrozenReceiverUniverse:
    """Freeze an order-invariant receiver axis without consulting context labels.

    ``observed_cell_type_ids`` must be computed from the complete sanitized root
    observation table without grouping, filtering, or selecting on context.  A
    predeclared axis may include receivers absent from the root data, but it must
    contain every observed root cell type.
    """

    if not isinstance(root_input_identity, SanitizedRawInputIdentity):
        raise TypeError("root_input_identity must be a SanitizedRawInputIdentity")
    root_input_identity._require_intact()
    observed = _canonical_ids(
        observed_cell_type_ids,
        field_name="observed_cell_type_ids",
    )
    if predeclared_receiver_ids is None:
        receivers = observed
        policy = (
            ReceiverUniverseSourcePolicy.ROOT_OBSERVED_CONTEXT_LABEL_BLIND_V1
        )
    else:
        receivers = _canonical_ids(
            predeclared_receiver_ids,
            field_name="predeclared_receiver_ids",
        )
        unknown = tuple(sorted(set(observed).difference(receivers)))
        if unknown:
            raise ContractError(
                "Observed root cell types fall outside the predeclared receiver axis",
                code="observed_cell_type_outside_predeclared_receiver_universe",
                field="observed_cell_type_ids",
                remediation=(
                    "Add every context-blind observed root cell type to the "
                    "predeclared receiver IDs"
                ),
            )
        policy = ReceiverUniverseSourcePolicy.EXPLICIT_PREDECLARED_V1
    return FrozenReceiverUniverse._from_root(
        root_input_identity,
        observed_cell_type_ids=observed,
        receiver_ids=receivers,
        source_policy=policy,
    )


@dataclass(frozen=True, slots=True, init=False)
class ReceiverTrainingSupportRecord:
    """Typed support only; this contract never contains a gate or a score."""

    receiver_universe_id: str
    outer_fold_id: str
    receiver_id: str
    training_cell_type_ids: tuple[str, ...]
    status: ReceiverTrainingSupportStatus
    reason_code: str | None
    support_record_id: str
    _universe: FrozenReceiverUniverse = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverTrainingSupportRecord is producer-owned; use "
            "assess_receiver_training_support()"
        )

    @classmethod
    def _from_training_axis(
        cls,
        universe: FrozenReceiverUniverse,
        *,
        outer_fold_id: str,
        receiver_id: str,
        training_cell_type_ids: tuple[str, ...],
    ) -> ReceiverTrainingSupportRecord:
        self = object.__new__(cls)
        is_observed = receiver_id in training_cell_type_ids
        object.__setattr__(self, "receiver_universe_id", universe.universe_id)
        object.__setattr__(self, "outer_fold_id", outer_fold_id)
        object.__setattr__(self, "receiver_id", receiver_id)
        object.__setattr__(self, "training_cell_type_ids", training_cell_type_ids)
        object.__setattr__(
            self,
            "status",
            (
                ReceiverTrainingSupportStatus.OBSERVED
                if is_observed
                else ReceiverTrainingSupportStatus.NOT_ESTIMABLE
            ),
        )
        object.__setattr__(self, "reason_code", None if is_observed else _ABSENT_REASON)
        object.__setattr__(self, "_universe", universe)
        object.__setattr__(self, "_producer_marker", _SUPPORT_PRODUCER)
        object.__setattr__(
            self,
            "support_record_id",
            stable_id(
                "receiver_training_support",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "outer_fold_id": self.outer_fold_id,
            "reason_code": self.reason_code,
            "receiver_id": self.receiver_id,
            "receiver_universe_id": self.receiver_universe_id,
            "status": self.status.value,
            "training_cell_type_ids": list(self.training_cell_type_ids),
        }

    def _require_intact(self) -> None:
        try:
            if not isinstance(self._universe, FrozenReceiverUniverse):
                raise TypeError("invalid receiver universe type")
            self._universe._require_intact()
            fold_id = _name(self.outer_fold_id, field_name="outer_fold_id")
            receiver_id = _name(self.receiver_id, field_name="receiver_id")
            training_ids = _canonical_ids(
                self.training_cell_type_ids,
                field_name="training_cell_type_ids",
                allow_empty=True,
            )
            if receiver_id not in self._universe.receiver_ids:
                raise ValueError("receiver is outside the frozen universe")
            if set(training_ids).difference(self._universe.receiver_ids):
                raise ValueError("training axis is outside the frozen universe")
            expected_status = (
                ReceiverTrainingSupportStatus.OBSERVED
                if receiver_id in training_ids
                else ReceiverTrainingSupportStatus.NOT_ESTIMABLE
            )
            expected_reason = (
                None
                if expected_status is ReceiverTrainingSupportStatus.OBSERVED
                else _ABSENT_REASON
            )
            expected_id = stable_id(
                "receiver_training_support",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _SUPPORT_PRODUCER
                and fold_id == self.outer_fold_id
                and training_ids == self.training_cell_type_ids
                and self.receiver_universe_id == self._universe.universe_id
                and isinstance(self.status, ReceiverTrainingSupportStatus)
                and self.status is expected_status
                and self.reason_code == expected_reason
                and self.support_record_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver training-support record failed integrity validation",
                code="receiver_training_support_integrity_violation",
                field="support_record_id",
                remediation=(
                    "Recompute training support from the intact frozen receiver "
                    "universe and outer-training cell-type IDs"
                ),
            ) from error
        if not valid:
            raise ContractError(
                "Receiver training-support record failed integrity validation",
                code="receiver_training_support_integrity_violation",
                field="support_record_id",
                remediation=(
                    "Recompute training support from the intact frozen receiver "
                    "universe and outer-training cell-type IDs"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        """Return a gate-free, score-free support row."""

        self._require_intact()
        return {
            "support_record_id": self.support_record_id,
            **self._identity_payload(),
        }


def assess_receiver_training_support(
    universe: FrozenReceiverUniverse,
    *,
    outer_fold_id: str,
    training_cell_type_ids: Sequence[str],
) -> tuple[ReceiverTrainingSupportRecord, ...]:
    """Return one complete, order-invariant support row per frozen receiver."""

    if not isinstance(universe, FrozenReceiverUniverse):
        raise TypeError("universe must be a FrozenReceiverUniverse")
    universe._require_intact()
    fold_id = _name(outer_fold_id, field_name="outer_fold_id")
    training_ids = _canonical_ids(
        training_cell_type_ids,
        field_name="training_cell_type_ids",
        allow_empty=True,
    )
    unknown = tuple(sorted(set(training_ids).difference(universe.receiver_ids)))
    if unknown:
        raise ContractError(
            "Outer-training cell types fall outside the frozen receiver universe",
            code="outer_training_cell_type_outside_receiver_universe",
            field="training_cell_type_ids",
            remediation="Use the complete run-level receiver universe for this fold",
        )
    return tuple(
        ReceiverTrainingSupportRecord._from_training_axis(
            universe,
            outer_fold_id=fold_id,
            receiver_id=receiver_id,
            training_cell_type_ids=training_ids,
        )
        for receiver_id in universe.receiver_ids
    )


__all__ = [
    "FrozenReceiverUniverse",
    "ReceiverTrainingSupportRecord",
    "ReceiverTrainingSupportStatus",
    "ReceiverUniverseReuseBinding",
    "ReceiverUniverseSourcePolicy",
    "assess_receiver_training_support",
    "bind_reused_receiver_universe",
    "freeze_receiver_universe",
]
