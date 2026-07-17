"""Frozen hypothesis universes and exact result-coverage contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from crychic.core import CommunicationMode, ContractError, stable_id

_SCHEMA_VERSION = "1.0.0"
_PRODUCER_MARKER = "crychic.inference.frozen_hypothesis_universe.v1"
_COVERAGE_PRODUCER_MARKER = "crychic.inference.hypothesis_coverage.v1"
_HIERARCHICAL_PROCEDURE_STATUS = (
    "frozen_treebh_benjamini_bogomolov_v1_g3_release_required"
)
_DENOMINATOR_POLICY = "all_frozen_hypotheses_including_prefiltered_v1"


class HypothesisRole(StrEnum):
    """Predeclared position in the primary/secondary hierarchy."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


class HypothesisFilterStage(StrEnum):
    """The only filtering stage accepted by this universe version."""

    PRE_FIT_OUTCOME_INDEPENDENT = "pre_fit_outcome_independent_v1"


class HypothesisPrefilterPolicy(StrEnum):
    """Reviewed sources for decisions made before effect fitting."""

    NONE = "none_predeclared_v1"
    EXTERNAL_RESOURCE = "external_resource_prefilter_v1"
    POOLED_CONTEXT_BLIND_SUPPORT = "pooled_context_label_blind_support_v1"


class HypothesisPrefilterStatus(StrEnum):
    """Frozen outcome of an allowed pre-fit filter."""

    INCLUDED = "included"
    FILTERED = "filtered_pre_fit"


class HypothesisCoverageStatus(StrEnum):
    """Availability of one frozen hypothesis in one result collection."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: str | None, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _filter_stage(value: object) -> HypothesisFilterStage:
    try:
        return HypothesisFilterStage(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ContractError(
            "Hypothesis filtering must be frozen before effect fitting and cannot "
            "use fitted effects, p-values, or q-values",
            code="post_fit_hypothesis_filtering_forbidden",
            field="filter_stage",
            remediation=(
                "Use the pre-fit outcome-independent stage and a reviewed policy"
            ),
        ) from error


def _prefilter_policy(value: object) -> HypothesisPrefilterPolicy:
    try:
        return HypothesisPrefilterPolicy(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ContractError(
            "Hypothesis filtering policy is not outcome-independent and reviewed",
            code="post_fit_hypothesis_filtering_forbidden",
            field="prefilter_policy",
            remediation=(
                "Use no filter, an external-resource filter, or pooled "
                "context-label-blind support"
            ),
        ) from error


@dataclass(frozen=True, slots=True, kw_only=True)
class HypothesisDeclaration:
    """One explicit hypothesis key and its pre-fit hierarchy/filter decision."""

    endpoint: str
    contrast_name: str
    receiver: str
    family_id: str
    mode: CommunicationMode
    role: HypothesisRole
    multiplicity_family: str
    parent_key: str | None = None
    filter_stage: HypothesisFilterStage = (
        HypothesisFilterStage.PRE_FIT_OUTCOME_INDEPENDENT
    )
    prefilter_policy: HypothesisPrefilterPolicy = HypothesisPrefilterPolicy.NONE
    prefilter_status: HypothesisPrefilterStatus = HypothesisPrefilterStatus.INCLUDED
    filter_reason_code: str | None = None
    hypothesis_key: str = field(init=False)
    hypothesis_id: str = field(init=False)
    declaration_id: str = field(init=False)

    def __post_init__(self) -> None:
        names = {
            field_name: _name(
                cast(str, getattr(self, field_name)),
                field_name=field_name,
            )
            for field_name in (
                "endpoint",
                "contrast_name",
                "receiver",
                "family_id",
                "multiplicity_family",
            )
        }
        mode = CommunicationMode(self.mode)
        role = HypothesisRole(self.role)
        parent_key = _optional_name(self.parent_key, field_name="parent_key")
        stage = _filter_stage(self.filter_stage)
        policy = _prefilter_policy(self.prefilter_policy)
        status = HypothesisPrefilterStatus(self.prefilter_status)
        if role is HypothesisRole.PRIMARY and parent_key is not None:
            raise ContractError(
                "Primary hypotheses cannot declare a hierarchy parent",
                code="invalid_primary_hypothesis_parent",
                field="parent_key",
                remediation="Remove the parent or declare a secondary hypothesis",
            )
        if role is HypothesisRole.SECONDARY and parent_key is None:
            raise ContractError(
                "Secondary hypotheses require an explicit primary parent key",
                code="secondary_hypothesis_parent_missing",
                field="parent_key",
                remediation="Bind the secondary hypothesis to its frozen primary",
            )
        reason = _optional_name(
            self.filter_reason_code,
            field_name="filter_reason_code",
        )
        if status is HypothesisPrefilterStatus.INCLUDED:
            if reason is not None:
                raise ValueError("included hypotheses cannot have a filter reason")
        else:
            if reason is None:
                raise ValueError("prefiltered hypotheses require a reason code")
            if policy is HypothesisPrefilterPolicy.NONE:
                raise ValueError("prefiltered hypotheses require a filtering policy")
        key_payload = {
            **names,
            "mode": mode.value,
            "role": role.value,
        }
        hypothesis_key = stable_id(
            "hypothesis_key",
            key_payload,
            schema_version=_SCHEMA_VERSION,
        )
        hypothesis_payload = {
            **key_payload,
            "parent_key": parent_key,
        }
        hypothesis_id = stable_id(
            "hypothesis",
            hypothesis_payload,
            schema_version=_SCHEMA_VERSION,
        )
        declaration_payload = {
            "hypothesis_id": hypothesis_id,
            "hypothesis_key": hypothesis_key,
            "filter_stage": stage.value,
            "prefilter_policy": policy.value,
            "prefilter_status": status.value,
            "filter_reason_code": reason,
        }
        for field_name, value in names.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "parent_key", parent_key)
        object.__setattr__(self, "filter_stage", stage)
        object.__setattr__(self, "prefilter_policy", policy)
        object.__setattr__(self, "prefilter_status", status)
        object.__setattr__(self, "filter_reason_code", reason)
        object.__setattr__(self, "hypothesis_key", hypothesis_key)
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(
            self,
            "declaration_id",
            stable_id(
                "hypothesis_declaration",
                declaration_payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _require_intact(self) -> None:
        try:
            repeated = HypothesisDeclaration(
                endpoint=self.endpoint,
                contrast_name=self.contrast_name,
                receiver=self.receiver,
                family_id=self.family_id,
                mode=self.mode,
                role=self.role,
                multiplicity_family=self.multiplicity_family,
                parent_key=self.parent_key,
                filter_stage=self.filter_stage,
                prefilter_policy=self.prefilter_policy,
                prefilter_status=self.prefilter_status,
                filter_reason_code=self.filter_reason_code,
            )
            valid = (
                self.hypothesis_key == repeated.hypothesis_key
                and self.hypothesis_id == repeated.hypothesis_id
                and self.declaration_id == repeated.declaration_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Hypothesis declaration failed integrity validation",
                code="hypothesis_declaration_integrity_violation",
                field="declaration_id",
                remediation="Recreate the declaration from its pre-fit fields",
            ) from error
        if not valid:
            raise ContractError(
                "Hypothesis declaration failed integrity validation",
                code="hypothesis_declaration_integrity_violation",
                field="declaration_id",
                remediation="Recreate the declaration from its pre-fit fields",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "declaration_id": self.declaration_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_key": self.hypothesis_key,
            "endpoint": self.endpoint,
            "contrast_name": self.contrast_name,
            "receiver": self.receiver,
            "family_id": self.family_id,
            "mode": self.mode.value,
            "role": self.role.value,
            "multiplicity_family": self.multiplicity_family,
            "parent_key": self.parent_key,
            "filter_stage": self.filter_stage.value,
            "prefilter_policy": self.prefilter_policy.value,
            "prefilter_status": self.prefilter_status.value,
            "filter_reason_code": self.filter_reason_code,
        }


def _validated_declarations(
    values: Sequence[HypothesisDeclaration],
) -> tuple[
    tuple[HypothesisDeclaration, ...],
    tuple[tuple[str, int], ...],
]:
    declarations = tuple(values)
    if not declarations or any(
        not isinstance(item, HypothesisDeclaration) for item in declarations
    ):
        raise ValueError("declarations must contain typed hypotheses")
    for declaration in declarations:
        declaration._require_intact()
    if len({item.hypothesis_key for item in declarations}) != len(declarations) or len(
        {item.hypothesis_id for item in declarations}
    ) != len(declarations):
        raise ContractError(
            "Frozen hypothesis declarations contain duplicate keys",
            code="duplicate_frozen_hypothesis",
            field="hypothesis_key",
            remediation="Declare each endpoint and analysis scope exactly once",
        )
    by_key = {item.hypothesis_key: item for item in declarations}
    for declaration in declarations:
        if declaration.role is HypothesisRole.PRIMARY:
            continue
        assert declaration.parent_key is not None
        parent = by_key.get(declaration.parent_key)
        if parent is None:
            raise ContractError(
                "Secondary hypothesis parent is absent from the frozen universe",
                code="dangling_hypothesis_parent",
                field="parent_key",
                remediation="Include the exact primary declaration in the universe",
            )
        if parent.role is not HypothesisRole.PRIMARY:
            raise ContractError(
                "Secondary hypothesis parent must be a primary hypothesis",
                code="invalid_hypothesis_parent_role",
                field="parent_key",
                remediation="Bind post-hoc hypotheses directly to a primary parent",
            )
        if (
            parent.receiver,
            parent.family_id,
            parent.mode,
        ) != (
            declaration.receiver,
            declaration.family_id,
            declaration.mode,
        ):
            raise ContractError(
                "Secondary hypothesis scope differs from its primary parent",
                code="hypothesis_parent_scope_mismatch",
                field="parent_key",
                remediation=(
                    "Use the primary hypothesis for the same receiver, family, and mode"
                ),
            )
        if (
            parent.prefilter_status is HypothesisPrefilterStatus.FILTERED
            and declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED
        ):
            raise ContractError(
                "A secondary hypothesis cannot bypass a prefiltered primary parent",
                code="hypothesis_prefilter_hierarchy_violation",
                field="prefilter_status",
                remediation=(
                    "Freeze the child as prefiltered or include its primary parent"
                ),
            )
    ordered = tuple(sorted(declarations, key=lambda item: item.hypothesis_id))
    counts = Counter(item.multiplicity_family for item in ordered)
    family_sizes = tuple(sorted(counts.items()))
    return ordered, family_sizes


@dataclass(frozen=True, slots=True, init=False)
class FrozenHypothesisUniverse:
    """Producer-owned immutable universe and multiplicity-family ledger."""

    universe_name: str
    declarations: tuple[HypothesisDeclaration, ...]
    hypothesis_ids: tuple[str, ...]
    multiplicity_family_sizes: tuple[tuple[str, int], ...]
    universe_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenHypothesisUniverse is producer-owned; use "
            "freeze_hypothesis_universe()"
        )

    @classmethod
    def _from_declarations(
        cls,
        declarations: Sequence[HypothesisDeclaration],
        *,
        universe_name: str,
    ) -> FrozenHypothesisUniverse:
        name = _name(universe_name, field_name="universe_name")
        ordered, family_sizes = _validated_declarations(declarations)
        self = object.__new__(cls)
        object.__setattr__(self, "universe_name", name)
        object.__setattr__(self, "declarations", ordered)
        object.__setattr__(
            self,
            "hypothesis_ids",
            tuple(item.hypothesis_id for item in ordered),
        )
        object.__setattr__(self, "multiplicity_family_sizes", family_sizes)
        object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
        object.__setattr__(
            self,
            "universe_id",
            stable_id(
                "frozen_hypothesis_universe",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    @property
    def q_value_release_allowed(self) -> bool:
        return False

    @property
    def hierarchical_procedure_status(self) -> str:
        return _HIERARCHICAL_PROCEDURE_STATUS

    @property
    def multiplicity_denominator(self) -> int:
        return len(self.hypothesis_ids)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_name": self.universe_name,
            "declaration_ids": [item.declaration_id for item in self.declarations],
            "hypothesis_ids": list(self.hypothesis_ids),
            "multiplicity_family_sizes": [
                [name, count] for name, count in self.multiplicity_family_sizes
            ],
            "multiplicity_denominator_policy": _DENOMINATOR_POLICY,
            "hierarchical_procedure_status": _HIERARCHICAL_PROCEDURE_STATUS,
            "q_value_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            name = _name(self.universe_name, field_name="universe_name")
            ordered, family_sizes = _validated_declarations(self.declarations)
            expected_ids = tuple(item.hypothesis_id for item in ordered)
            expected_id = stable_id(
                "frozen_hypothesis_universe",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and name == self.universe_name
                and ordered == self.declarations
                and expected_ids == self.hypothesis_ids
                and family_sizes == self.multiplicity_family_sizes
                and expected_id == self.universe_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen hypothesis universe failed integrity validation",
                code="frozen_hypothesis_universe_integrity_violation",
                field="universe_id",
                remediation="Refreeze the complete pre-fit hypothesis declarations",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen hypothesis universe failed integrity validation",
                code="frozen_hypothesis_universe_integrity_violation",
                field="universe_id",
                remediation="Refreeze the complete pre-fit hypothesis declarations",
            )

    def declaration_for(self, hypothesis_id: str) -> HypothesisDeclaration:
        """Return the exact frozen declaration or fail on an unknown ID."""

        self._require_intact()
        identifier = _name(hypothesis_id, field_name="hypothesis_id")
        matched = tuple(
            item for item in self.declarations if item.hypothesis_id == identifier
        )
        if len(matched) != 1:
            raise ContractError(
                "Hypothesis is absent from the frozen universe",
                code="hypothesis_not_in_frozen_universe",
                field="hypothesis_id",
                remediation="Use an exact hypothesis ID from this universe",
            )
        return matched[0]

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "universe_id": self.universe_id,
            **self._identity_payload(),
            "multiplicity_denominator": self.multiplicity_denominator,
            "declarations": [item.to_dict() for item in self.declarations],
        }


def freeze_hypothesis_universe(
    declarations: Sequence[HypothesisDeclaration],
    *,
    universe_name: str,
) -> FrozenHypothesisUniverse:
    """Freeze a complete order-invariant hierarchy before effect fitting."""

    return FrozenHypothesisUniverse._from_declarations(
        declarations,
        universe_name=universe_name,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class HypothesisCoverageRecord:
    """One status row for one exact frozen hypothesis."""

    hypothesis_id: str
    status: HypothesisCoverageStatus
    reason_code: str | None = None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        hypothesis_id = _name(self.hypothesis_id, field_name="hypothesis_id")
        status = HypothesisCoverageStatus(self.status)
        reason = _optional_name(self.reason_code, field_name="reason_code")
        if status is HypothesisCoverageStatus.OBSERVED:
            if reason is not None:
                raise ValueError("observed coverage cannot have a reason code")
        elif reason is None:
            raise ValueError("unavailable coverage requires a reason code")
        payload = {
            "hypothesis_id": hypothesis_id,
            "status": status.value,
            "reason_code": reason,
        }
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "hypothesis_coverage_record",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _require_intact(self) -> None:
        try:
            repeated = HypothesisCoverageRecord(
                hypothesis_id=self.hypothesis_id,
                status=self.status,
                reason_code=self.reason_code,
            )
            valid = self.record_id == repeated.record_id
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Hypothesis coverage record failed integrity validation",
                code="hypothesis_coverage_record_integrity_violation",
                field="record_id",
                remediation="Recreate the typed coverage row",
            ) from error
        if not valid:
            raise ContractError(
                "Hypothesis coverage record failed integrity validation",
                code="hypothesis_coverage_record_integrity_violation",
                field="record_id",
                remediation="Recreate the typed coverage row",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "record_id": self.record_id,
            "hypothesis_id": self.hypothesis_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
        }


def _validated_coverage(
    universe: FrozenHypothesisUniverse,
    records: Sequence[HypothesisCoverageRecord],
) -> tuple[HypothesisCoverageRecord, ...]:
    universe._require_intact()
    supplied = tuple(records)
    if not supplied or any(
        not isinstance(item, HypothesisCoverageRecord) for item in supplied
    ):
        raise ValueError("records must contain typed hypothesis coverage rows")
    for record in supplied:
        record._require_intact()
    identifiers = tuple(item.hypothesis_id for item in supplied)
    if len(set(identifiers)) != len(identifiers):
        raise ContractError(
            "Hypothesis coverage contains duplicate rows",
            code="duplicate_hypothesis_coverage",
            field="hypothesis_id",
            remediation="Emit exactly one status row per frozen hypothesis",
        )
    expected = set(universe.hypothesis_ids)
    observed = set(identifiers)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ContractError(
            "Hypothesis coverage does not equal the frozen universe",
            code=(
                "missing_hypothesis_coverage"
                if missing and not extra
                else (
                    "extra_hypothesis_coverage"
                    if extra and not missing
                    else "hypothesis_coverage_universe_mismatch"
                )
            ),
            field="hypothesis_id",
            remediation=(
                "Retain one observed, not-estimable, or failed row for every "
                "frozen hypothesis"
            ),
        )
    by_id = {item.hypothesis_id: item for item in supplied}
    for declaration in universe.declarations:
        record = by_id[declaration.hypothesis_id]
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED and (
            record.status is not HypothesisCoverageStatus.NOT_ESTIMABLE
            or record.reason_code != declaration.filter_reason_code
        ):
            raise ContractError(
                "Prefiltered hypothesis must remain an exact not-estimable row",
                code="prefiltered_hypothesis_coverage_mismatch",
                field="hypothesis_id",
                remediation=(
                    "Keep the frozen filter reason and do not test or drop the row"
                ),
            )
    return tuple(sorted(supplied, key=lambda item: item.hypothesis_id))


@dataclass(frozen=True, slots=True, init=False)
class FrozenHypothesisCoverage:
    """Producer-owned proof of exact, denominator-preserving result coverage."""

    universe_id: str
    records: tuple[HypothesisCoverageRecord, ...]
    multiplicity_family_sizes: tuple[tuple[str, int], ...]
    n_observed: int
    n_not_estimable: int
    n_failed: int
    coverage_id: str
    _universe: FrozenHypothesisUniverse
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenHypothesisCoverage is producer-owned; use "
            "validate_frozen_hypothesis_coverage()"
        )

    @classmethod
    def _from_records(
        cls,
        universe: FrozenHypothesisUniverse,
        records: Sequence[HypothesisCoverageRecord],
    ) -> FrozenHypothesisCoverage:
        ordered = _validated_coverage(universe, records)
        counts = Counter(item.status for item in ordered)
        self = object.__new__(cls)
        object.__setattr__(self, "universe_id", universe.universe_id)
        object.__setattr__(self, "records", ordered)
        object.__setattr__(
            self,
            "multiplicity_family_sizes",
            universe.multiplicity_family_sizes,
        )
        object.__setattr__(
            self,
            "n_observed",
            counts[HypothesisCoverageStatus.OBSERVED],
        )
        object.__setattr__(
            self,
            "n_not_estimable",
            counts[HypothesisCoverageStatus.NOT_ESTIMABLE],
        )
        object.__setattr__(
            self,
            "n_failed",
            counts[HypothesisCoverageStatus.FAILED],
        )
        object.__setattr__(self, "_universe", universe)
        object.__setattr__(self, "_producer_marker", _COVERAGE_PRODUCER_MARKER)
        object.__setattr__(
            self,
            "coverage_id",
            stable_id(
                "frozen_hypothesis_coverage",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    @property
    def complete(self) -> bool:
        return True

    @property
    def q_value_release_allowed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "record_ids": [item.record_id for item in self.records],
            "multiplicity_family_sizes": [
                [name, count] for name, count in self.multiplicity_family_sizes
            ],
            "n_observed": self.n_observed,
            "n_not_estimable": self.n_not_estimable,
            "n_failed": self.n_failed,
            "multiplicity_denominator_policy": _DENOMINATOR_POLICY,
            "hierarchical_procedure_status": _HIERARCHICAL_PROCEDURE_STATUS,
            "q_value_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            ordered = _validated_coverage(self._universe, self.records)
            counts = Counter(item.status for item in ordered)
            expected_id = stable_id(
                "frozen_hypothesis_coverage",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _COVERAGE_PRODUCER_MARKER
                and self.universe_id == self._universe.universe_id
                and ordered == self.records
                and self.multiplicity_family_sizes
                == self._universe.multiplicity_family_sizes
                and self.n_observed
                == counts[HypothesisCoverageStatus.OBSERVED]
                and self.n_not_estimable
                == counts[HypothesisCoverageStatus.NOT_ESTIMABLE]
                and self.n_failed == counts[HypothesisCoverageStatus.FAILED]
                and expected_id == self.coverage_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Frozen hypothesis coverage failed integrity validation",
                code="frozen_hypothesis_coverage_integrity_violation",
                field="coverage_id",
                remediation="Revalidate complete typed rows against the universe",
            ) from error
        if not valid:
            raise ContractError(
                "Frozen hypothesis coverage failed integrity validation",
                code="frozen_hypothesis_coverage_integrity_violation",
                field="coverage_id",
                remediation="Revalidate complete typed rows against the universe",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "coverage_id": self.coverage_id,
            **self._identity_payload(),
            "complete": True,
            "records": [item.to_dict() for item in self.records],
        }


def validate_frozen_hypothesis_coverage(
    universe: FrozenHypothesisUniverse,
    records: Sequence[HypothesisCoverageRecord],
) -> FrozenHypothesisCoverage:
    """Require exactly one typed status row for every frozen hypothesis."""

    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be a FrozenHypothesisUniverse")
    return FrozenHypothesisCoverage._from_records(universe, records)


__all__ = [
    "FrozenHypothesisCoverage",
    "FrozenHypothesisUniverse",
    "HypothesisCoverageRecord",
    "HypothesisCoverageStatus",
    "HypothesisDeclaration",
    "HypothesisFilterStage",
    "HypothesisPrefilterPolicy",
    "HypothesisPrefilterStatus",
    "HypothesisRole",
    "freeze_hypothesis_universe",
    "validate_frozen_hypothesis_coverage",
]
