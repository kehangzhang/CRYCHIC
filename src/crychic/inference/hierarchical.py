"""Frozen two-level TreeBH/Benjamini-Bogomolov candidate adjustment.

The module computes auditable candidate q-values for one primary omnibus layer
and one secondary post-hoc layer.  It does not treat selective-FDR q-values as
pooled leaf-FDR q-values. Formal q-value release requires a producer-owned G3
calibration gate bound to the exact universe and procedure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from crychic.core import CommunicationMode, ContractError, stable_id

from .hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisCoverageStatus,
    HypothesisDeclaration,
    HypothesisPrefilterStatus,
    HypothesisRole,
)

if TYPE_CHECKING:
    from .full_pipeline import G3FrequencyCalibrationGate

_SCHEMA_VERSION = "1.0.0"
_ALPHA = 0.05
_PRIMARY_PROCEDURE = "benjamini_hochberg_v1"
_SECONDARY_PROCEDURE = "benjamini_bogomolov_selected_family_bh_v1"
_SECONDARY_LEVEL_RULE = "alpha_times_selected_primary_count_over_primary_count_v1"
_Q_VALUE_SCOPE = "selective_fdr_not_pooled_leaf_fdr"
_DENOMINATOR_POLICY = "all_frozen_hypotheses_nonobserved_effective_p_one_v1"
_SPEC_MARKER = "crychic.inference.hierarchical_fdr_spec.v1"
_RESULT_MARKER = "crychic.inference.hierarchical_fdr_collection.v1"
_HYPOTHESIS_RESULT_MARKER = "crychic.inference.hierarchical_fdr_record.v1"
_PRIMARY_ENDPOINT = "driver_family_receiver_context_omnibus_v1"
_SECONDARY_ENDPOINT = "family_common_integrated_lr_context_effect_v1"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_name(value: object | None, *, field_name: str) -> str | None:
    return None if value is None else _name(value, field_name=field_name)


def _probability(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a finite value in [0, 1]") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be a finite value in [0, 1]")
    return 0.0 if result == 0.0 else result


def _leq(left: float, right: float) -> bool:
    return left <= right or math.isclose(
        left,
        right,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )


class HierarchicalFDRReleaseStatus(StrEnum):
    """Whether candidate q-values cross the formal release boundary."""

    RELEASED = "released"
    CANDIDATE_ONLY = "candidate_only"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True, init=False)
class FrozenHierarchicalFDRSpec:
    """Producer-owned fixed two-level selective-FDR procedure."""

    alpha: float
    primary_procedure: str
    secondary_procedure: str
    secondary_level_rule: str
    q_value_scope: str
    denominator_policy: str
    procedure_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenHierarchicalFDRSpec is producer-owned; use "
            "freeze_hierarchical_fdr_spec()"
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "primary_procedure": self.primary_procedure,
            "secondary_procedure": self.secondary_procedure,
            "secondary_level_rule": self.secondary_level_rule,
            "q_value_scope": self.q_value_scope,
            "denominator_policy": self.denominator_policy,
        }

    def _require_intact(self) -> None:
        try:
            expected_id = stable_id(
                "hierarchical_fdr_procedure",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _SPEC_MARKER
                and self.alpha == _ALPHA
                and self.primary_procedure == _PRIMARY_PROCEDURE
                and self.secondary_procedure == _SECONDARY_PROCEDURE
                and self.secondary_level_rule == _SECONDARY_LEVEL_RULE
                and self.q_value_scope == _Q_VALUE_SCOPE
                and self.denominator_policy == _DENOMINATOR_POLICY
                and self.procedure_id == expected_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Hierarchical FDR specification failed integrity validation",
                code="hierarchical_fdr_spec_integrity_violation",
                field="procedure_id",
                remediation="Refreeze the fixed hierarchical FDR specification",
            ) from error
        if not valid:
            raise ContractError(
                "Hierarchical FDR specification failed integrity validation",
                code="hierarchical_fdr_spec_integrity_violation",
                field="procedure_id",
                remediation="Refreeze the fixed hierarchical FDR specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"procedure_id": self.procedure_id, **self._identity_payload()}


def freeze_hierarchical_fdr_spec() -> FrozenHierarchicalFDRSpec:
    """Return the only candidate procedure implemented by this module."""

    values: dict[str, object] = {
        "alpha": _ALPHA,
        "primary_procedure": _PRIMARY_PROCEDURE,
        "secondary_procedure": _SECONDARY_PROCEDURE,
        "secondary_level_rule": _SECONDARY_LEVEL_RULE,
        "q_value_scope": _Q_VALUE_SCOPE,
        "denominator_policy": _DENOMINATOR_POLICY,
    }
    self = object.__new__(FrozenHierarchicalFDRSpec)
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_producer_marker", _SPEC_MARKER)
    object.__setattr__(
        self,
        "procedure_id",
        stable_id(
            "hierarchical_fdr_procedure",
            values,
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


@dataclass(frozen=True, slots=True, kw_only=True)
class HypothesisPValueRecord:
    """One typed p-value/status row from an upstream full-pipeline result."""

    hypothesis_id: str
    source_result_id: str
    status: HypothesisCoverageStatus
    p_value: float | None = None
    reason_code: str | None = None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        hypothesis_id = _name(self.hypothesis_id, field_name="hypothesis_id")
        source_result_id = _name(
            self.source_result_id,
            field_name="source_result_id",
        )
        status = HypothesisCoverageStatus(self.status)
        reason = _optional_name(self.reason_code, field_name="reason_code")
        if status is HypothesisCoverageStatus.OBSERVED:
            if self.p_value is None:
                raise ValueError("observed p-value record requires p_value")
            p_value = _probability(self.p_value, field_name="p_value")
            if reason is not None:
                raise ValueError("observed p-value record cannot have a reason code")
        else:
            if self.p_value is not None:
                raise ValueError("non-observed p-value record requires p_value=None")
            if reason is None:
                raise ValueError("non-observed p-value record requires a reason code")
            p_value = None
        payload = {
            "hypothesis_id": hypothesis_id,
            "source_result_id": source_result_id,
            "status": status.value,
            "p_value": p_value,
            "reason_code": reason,
            "effective_p_value": 1.0 if p_value is None else p_value,
        }
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "source_result_id", source_result_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "hypothesis_p_value_record",
                payload,
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def effective_p_value(self) -> float:
        """Return one for typed unavailable rows without exposing a fake p-value."""

        return 1.0 if self.p_value is None else self.p_value

    def _require_intact(self) -> None:
        try:
            repeated = HypothesisPValueRecord(
                hypothesis_id=self.hypothesis_id,
                source_result_id=self.source_result_id,
                status=self.status,
                p_value=self.p_value,
                reason_code=self.reason_code,
            )
            valid = self.record_id == repeated.record_id
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Hypothesis p-value record failed integrity validation",
                code="hypothesis_p_value_record_integrity_violation",
                field="record_id",
                remediation="Recreate the typed p-value record",
            ) from error
        if not valid:
            raise ContractError(
                "Hypothesis p-value record failed integrity validation",
                code="hypothesis_p_value_record_integrity_violation",
                field="record_id",
                remediation="Recreate the typed p-value record",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "record_id": self.record_id,
            "hypothesis_id": self.hypothesis_id,
            "source_result_id": self.source_result_id,
            "status": self.status.value,
            "p_value": self.p_value,
            "effective_p_value": self.effective_p_value,
            "reason_code": self.reason_code,
        }


def _bh_q_values(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted = [1.0] * count
    running = 1.0
    for offset in range(count - 1, -1, -1):
        rank = offset + 1
        running = min(running, ordered[offset][1] * count / rank, 1.0)
        adjusted[offset] = running
    return {
        hypothesis_id: adjusted[index]
        for index, (hypothesis_id, _) in enumerate(ordered)
    }


def _bh_rejections(p_values: Mapping[str, float], level: float) -> frozenset[str]:
    adjusted = _bh_q_values(p_values)
    return frozenset(
        hypothesis_id
        for hypothesis_id, q_value in adjusted.items()
        if _leq(q_value, level)
    )


@dataclass(frozen=True, slots=True)
class _Hierarchy:
    declarations_by_id: Mapping[str, HypothesisDeclaration]
    primary_by_id: Mapping[str, HypothesisDeclaration]
    parent_id_by_child_id: Mapping[str, str]
    children_by_parent_id: Mapping[str, tuple[str, ...]]


def _validate_v1_universe_scope(
    declarations: Sequence[HypothesisDeclaration],
) -> None:
    primary = tuple(
        item for item in declarations if item.role is HypothesisRole.PRIMARY
    )
    secondary = tuple(
        item for item in declarations if item.role is HypothesisRole.SECONDARY
    )
    if any(item.endpoint != _PRIMARY_ENDPOINT for item in primary) or any(
        item.endpoint != _SECONDARY_ENDPOINT for item in secondary
    ):
        raise ContractError(
            "Hierarchical FDR v1 only accepts its frozen omnibus/effect endpoints",
            code="hierarchical_fdr_v1_endpoint_scope_mismatch",
            field="endpoint",
            remediation=(
                "Use the v1 driver-family omnibus primary endpoint and the "
                "family-common integrated LR context-effect secondary endpoint"
            ),
        )
    if any(item.mode is not CommunicationMode.STATE for item in declarations):
        raise ContractError(
            "Hierarchical FDR v1 is calibrated only for state-mode hypotheses",
            code="hierarchical_fdr_v1_mode_scope_mismatch",
            field="mode",
            remediation="Freeze a state-only v1 hypothesis universe",
        )
    for role, layer in (
        (HypothesisRole.PRIMARY, primary),
        (HypothesisRole.SECONDARY, secondary),
    ):
        if layer and len({item.multiplicity_family for item in layer}) != 1:
            raise ContractError(
                "Each hierarchical FDR v1 layer requires one multiplicity family",
                code="hierarchical_fdr_v1_multiplicity_scope_mismatch",
                field="multiplicity_family",
                remediation=(
                    f"Freeze all {role.value} hypotheses in one layer-wide "
                    "multiplicity family"
                ),
            )


def _hierarchy(universe: FrozenHypothesisUniverse) -> _Hierarchy:
    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be a FrozenHypothesisUniverse")
    universe.to_dict()
    _validate_v1_universe_scope(universe.declarations)
    declarations = {item.hypothesis_id: item for item in universe.declarations}
    primary = {
        item.hypothesis_id: item
        for item in universe.declarations
        if item.role is HypothesisRole.PRIMARY
    }
    if not primary:
        raise ContractError(
            "Hierarchical FDR requires at least one primary hypothesis",
            code="hierarchical_fdr_primary_layer_empty",
            field="role",
            remediation="Freeze primary omnibus hypotheses before post-hoc tests",
        )
    primary_by_key = {item.hypothesis_key: item for item in primary.values()}
    parent_by_child: dict[str, str] = {}
    children: defaultdict[str, list[str]] = defaultdict(list)
    for declaration in universe.declarations:
        if declaration.role is HypothesisRole.PRIMARY:
            continue
        if (
            declaration.parent_key is None
            or declaration.parent_key not in primary_by_key
        ):
            raise ContractError(
                "Secondary hypothesis lacks an exact primary parent",
                code="hierarchical_fdr_secondary_parent_invalid",
                field="parent_key",
                remediation="Refreeze an exact two-level primary/secondary hierarchy",
            )
        parent_id = primary_by_key[declaration.parent_key].hypothesis_id
        parent_by_child[declaration.hypothesis_id] = parent_id
        children[parent_id].append(declaration.hypothesis_id)
    return _Hierarchy(
        declarations_by_id=declarations,
        primary_by_id=primary,
        parent_id_by_child_id=parent_by_child,
        children_by_parent_id={
            parent_id: tuple(sorted(values)) for parent_id, values in children.items()
        },
    )


def _validated_records(
    universe: FrozenHypothesisUniverse,
    records: Sequence[HypothesisPValueRecord],
) -> tuple[HypothesisPValueRecord, ...]:
    supplied = tuple(records)
    if not supplied or any(
        not isinstance(item, HypothesisPValueRecord) for item in supplied
    ):
        raise ValueError("records must contain typed hypothesis p-value rows")
    for record in supplied:
        record._require_intact()
    identifiers = tuple(item.hypothesis_id for item in supplied)
    if len(set(identifiers)) != len(identifiers):
        raise ContractError(
            "Hypothesis p-value input contains duplicate rows",
            code="duplicate_hypothesis_p_value",
            field="hypothesis_id",
            remediation="Emit exactly one p-value/status row per frozen hypothesis",
        )
    expected = set(universe.hypothesis_ids)
    observed = set(identifiers)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ContractError(
            "Hypothesis p-value coverage does not equal the frozen universe",
            code=(
                "missing_hypothesis_p_value"
                if missing and not extra
                else (
                    "extra_hypothesis_p_value"
                    if extra and not missing
                    else "hypothesis_p_value_universe_mismatch"
                )
            ),
            field="hypothesis_id",
            remediation="Retain one typed row for every frozen hypothesis",
        )
    by_id = {item.hypothesis_id: item for item in supplied}
    for declaration in universe.declarations:
        record = by_id[declaration.hypothesis_id]
        if declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED and (
            record.status is not HypothesisCoverageStatus.NOT_ESTIMABLE
            or record.reason_code != declaration.filter_reason_code
        ):
            raise ContractError(
                "Prefiltered hypothesis must retain its exact frozen reason",
                code="prefiltered_hypothesis_p_value_mismatch",
                field="hypothesis_id",
                remediation="Use p=None, not_estimable, and the frozen filter reason",
            )
    return tuple(sorted(supplied, key=lambda item: item.hypothesis_id))


def _decisions_at_level(
    *,
    hierarchy: _Hierarchy,
    effective_p: Mapping[str, float],
    level: float,
) -> tuple[frozenset[str], frozenset[str], float]:
    primary_p = {
        hypothesis_id: effective_p[hypothesis_id]
        for hypothesis_id in hierarchy.primary_by_id
    }
    selected_primary = _bh_rejections(primary_p, level)
    primary_count = len(primary_p)
    child_level = (
        level * len(selected_primary) / primary_count if primary_count else 0.0
    )
    selected_secondary: set[str] = set()
    for parent_id in selected_primary:
        child_ids = hierarchy.children_by_parent_id.get(parent_id, ())
        child_p = {
            hypothesis_id: effective_p[hypothesis_id] for hypothesis_id in child_ids
        }
        selected_secondary.update(_bh_rejections(child_p, child_level))
    return selected_primary, frozenset(selected_secondary), child_level


def _candidate_q_values(
    hierarchy: _Hierarchy,
    effective_p: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    primary_p = {
        hypothesis_id: effective_p[hypothesis_id]
        for hypothesis_id in hierarchy.primary_by_id
    }
    primary_q = _bh_q_values(primary_p)
    primary_count = len(primary_p)
    breakpoints = {0.0, 1.0, *primary_q.values()}
    for child_ids in hierarchy.children_by_parent_id.values():
        within_q = _bh_q_values(
            {hypothesis_id: effective_p[hypothesis_id] for hypothesis_id in child_ids}
        )
        for q_value in within_q.values():
            for selected_count in range(1, primary_count + 1):
                breakpoints.add(min(1.0, q_value * primary_count / selected_count))
    secondary_ids = set(hierarchy.parent_id_by_child_id)
    secondary_q = {hypothesis_id: 1.0 for hypothesis_id in secondary_ids}
    unresolved = set(secondary_ids)
    for level in sorted(breakpoints):
        if not unresolved:
            break
        _, selected_secondary, _ = _decisions_at_level(
            hierarchy=hierarchy,
            effective_p=effective_p,
            level=level,
        )
        newly_selected = unresolved.intersection(selected_secondary)
        for hypothesis_id in newly_selected:
            secondary_q[hypothesis_id] = level
        unresolved.difference_update(newly_selected)
    return primary_q, secondary_q


def _validated_calibration_gate(
    calibration_gate: object | None,
) -> G3FrequencyCalibrationGate | None:
    from .full_pipeline import G3FrequencyCalibrationGate

    if calibration_gate is None:
        return None
    if not isinstance(calibration_gate, G3FrequencyCalibrationGate):
        raise TypeError(
            "calibration_gate must be a producer-owned "
            "G3FrequencyCalibrationGate, never a boolean"
        )
    calibration_gate._require_intact()
    return calibration_gate


def _release_boundary(
    *,
    universe: FrozenHypothesisUniverse,
    spec: FrozenHierarchicalFDRSpec,
    records: Sequence[HypothesisPValueRecord],
    calibration_gate: G3FrequencyCalibrationGate | None,
) -> tuple[
    HierarchicalFDRReleaseStatus,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    gate = _validated_calibration_gate(calibration_gate)
    if gate is not None and gate.hypothesis_universe_id != universe.universe_id:
        raise ContractError(
            "G3 calibration gate is bound to a different hypothesis universe",
            code="hierarchical_fdr_gate_universe_mismatch",
            field="hypothesis_universe_id",
            remediation="Calibrate the exact frozen hypothesis universe",
        )
    if gate is not None and gate.hierarchical_procedure_id != spec.procedure_id:
        raise ContractError(
            "G3 calibration gate is bound to a different FDR procedure",
            code="hierarchical_fdr_gate_procedure_mismatch",
            field="hierarchical_procedure_id",
            remediation="Calibrate the exact frozen hierarchical FDR procedure",
        )
    gate_lineage = (
        (None, None, None)
        if gate is None
        else (gate.gate_id, gate.evidence_id, gate.protocol_id)
    )
    unavailable_included = any(
        declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED
        and record.status is not HypothesisCoverageStatus.OBSERVED
        for declaration, record in (
            (universe.declaration_for(item.hypothesis_id), item) for item in records
        )
    )
    if unavailable_included:
        return (
            HierarchicalFDRReleaseStatus.BLOCKED,
            "included_hypothesis_p_value_unavailable",
            *gate_lineage,
        )
    if gate is None:
        return (
            HierarchicalFDRReleaseStatus.CANDIDATE_ONLY,
            "hierarchical_fdr_calibration_gate_missing",
            None,
            None,
            None,
        )
    if not gate.hierarchical_q_release_allowed:
        return (
            HierarchicalFDRReleaseStatus.BLOCKED,
            gate.hierarchical_reason_code
            or gate.reason_code
            or "g3_hierarchical_q_release_not_allowed",
            *gate_lineage,
        )
    return (
        HierarchicalFDRReleaseStatus.RELEASED,
        None,
        *gate_lineage,
    )


@dataclass(frozen=True, slots=True, init=False)
class HierarchicalFDRHypothesisResult:
    """Producer-owned candidate and formally released result for one hypothesis."""

    hypothesis_id: str
    p_value_record_id: str
    role: HypothesisRole
    parent_hypothesis_id: str | None
    status: HypothesisCoverageStatus
    reason_code: str | None
    effective_p_value: float
    candidate_q_value: float
    q_value: float | None
    candidate_rejected_at_alpha: bool
    rejected_at_alpha: bool | None
    candidate_parent_selected_at_alpha: bool | None
    parent_selected_at_alpha: bool | None
    secondary_test_level: float | None
    result_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError("HierarchicalFDRHypothesisResult is producer-owned")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "p_value_record_id": self.p_value_record_id,
            "role": self.role.value,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "effective_p_value": self.effective_p_value,
            "candidate_q_value": self.candidate_q_value,
            "q_value": self.q_value,
            "candidate_rejected_at_alpha": self.candidate_rejected_at_alpha,
            "rejected_at_alpha": self.rejected_at_alpha,
            "candidate_parent_selected_at_alpha": (
                self.candidate_parent_selected_at_alpha
            ),
            "parent_selected_at_alpha": self.parent_selected_at_alpha,
            "secondary_test_level": self.secondary_test_level,
        }

    def _require_intact(self) -> None:
        try:
            expected_id = stable_id(
                "hierarchical_fdr_hypothesis_result",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _HYPOTHESIS_RESULT_MARKER
                and self.result_id == expected_id
                and 0.0 <= self.effective_p_value <= 1.0
                and 0.0 <= self.candidate_q_value <= 1.0
                and (self.q_value is None or 0.0 <= self.q_value <= 1.0)
                and isinstance(self.candidate_rejected_at_alpha, bool)
                and (
                    self.rejected_at_alpha is None
                    or isinstance(self.rejected_at_alpha, bool)
                )
                and (
                    self.candidate_parent_selected_at_alpha is None
                    or isinstance(self.candidate_parent_selected_at_alpha, bool)
                )
                and (
                    self.parent_selected_at_alpha is None
                    or isinstance(self.parent_selected_at_alpha, bool)
                )
                and self.candidate_rejected_at_alpha
                == _leq(self.candidate_q_value, _ALPHA)
                and (self.q_value is None) == (self.rejected_at_alpha is None)
                and (self.q_value is None or self.q_value == self.candidate_q_value)
                and (
                    self.rejected_at_alpha is None
                    or self.rejected_at_alpha == self.candidate_rejected_at_alpha
                )
                and (
                    self.parent_selected_at_alpha is None
                    or self.parent_selected_at_alpha
                    == self.candidate_parent_selected_at_alpha
                )
                and (
                    self.role is not HypothesisRole.PRIMARY
                    or (
                        self.candidate_parent_selected_at_alpha is None
                        and self.parent_selected_at_alpha is None
                    )
                )
                and (
                    self.role is not HypothesisRole.SECONDARY
                    or isinstance(self.candidate_parent_selected_at_alpha, bool)
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Hierarchical FDR hypothesis result failed integrity validation",
                code="hierarchical_fdr_record_integrity_violation",
                field="result_id",
                remediation="Recompute the complete hierarchical FDR collection",
            ) from error
        if not valid:
            raise ContractError(
                "Hierarchical FDR hypothesis result failed integrity validation",
                code="hierarchical_fdr_record_integrity_violation",
                field="result_id",
                remediation="Recompute the complete hierarchical FDR collection",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"result_id": self.result_id, **self._identity_payload()}


def _result_record(**values: object) -> HierarchicalFDRHypothesisResult:
    self = object.__new__(HierarchicalFDRHypothesisResult)
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_producer_marker", _HYPOTHESIS_RESULT_MARKER)
    object.__setattr__(
        self,
        "result_id",
        stable_id(
            "hierarchical_fdr_hypothesis_result",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


@dataclass(frozen=True, slots=True)
class _Evaluation:
    records: tuple[HierarchicalFDRHypothesisResult, ...]
    n_primary: int
    n_primary_selected: int
    secondary_test_level: float
    release_status: HierarchicalFDRReleaseStatus
    release_reason_code: str | None
    calibration_gate_id: str | None
    calibration_evidence_id: str | None
    calibration_protocol_id: str | None


def _evaluate(
    *,
    universe: FrozenHypothesisUniverse,
    spec: FrozenHierarchicalFDRSpec,
    p_value_records: Sequence[HypothesisPValueRecord],
    calibration_gate: G3FrequencyCalibrationGate | None,
) -> _Evaluation:
    spec._require_intact()
    hierarchy = _hierarchy(universe)
    ordered = _validated_records(universe, p_value_records)
    by_id = {item.hypothesis_id: item for item in ordered}
    effective_p = {
        hypothesis_id: record.effective_p_value
        for hypothesis_id, record in by_id.items()
    }
    primary_q, secondary_q = _candidate_q_values(hierarchy, effective_p)
    selected_primary, selected_secondary, child_level = _decisions_at_level(
        hierarchy=hierarchy,
        effective_p=effective_p,
        level=spec.alpha,
    )
    (
        release_status,
        release_reason,
        calibration_gate_id,
        calibration_evidence_id,
        calibration_protocol_id,
    ) = _release_boundary(
        universe=universe,
        spec=spec,
        records=ordered,
        calibration_gate=calibration_gate,
    )
    release_allowed = release_status is HierarchicalFDRReleaseStatus.RELEASED
    results: list[HierarchicalFDRHypothesisResult] = []
    for declaration in universe.declarations:
        record = by_id[declaration.hypothesis_id]
        if declaration.role is HypothesisRole.PRIMARY:
            candidate_q = primary_q[declaration.hypothesis_id]
            parent_id = None
            candidate_parent_selected: bool | None = None
            secondary_level: float | None = None
            candidate_rejected = declaration.hypothesis_id in selected_primary
        else:
            candidate_q = secondary_q[declaration.hypothesis_id]
            parent_id = hierarchy.parent_id_by_child_id[declaration.hypothesis_id]
            candidate_parent_selected = parent_id in selected_primary
            secondary_level = child_level
            candidate_rejected = declaration.hypothesis_id in selected_secondary
        formal_decision_available = (
            release_allowed and record.status is HypothesisCoverageStatus.OBSERVED
        )
        results.append(
            _result_record(
                hypothesis_id=declaration.hypothesis_id,
                p_value_record_id=record.record_id,
                role=declaration.role,
                parent_hypothesis_id=parent_id,
                status=record.status,
                reason_code=record.reason_code,
                effective_p_value=record.effective_p_value,
                candidate_q_value=candidate_q,
                q_value=candidate_q if formal_decision_available else None,
                candidate_rejected_at_alpha=candidate_rejected,
                rejected_at_alpha=(
                    candidate_rejected if formal_decision_available else None
                ),
                candidate_parent_selected_at_alpha=candidate_parent_selected,
                parent_selected_at_alpha=(
                    candidate_parent_selected if formal_decision_available else None
                ),
                secondary_test_level=secondary_level,
            )
        )
    return _Evaluation(
        records=tuple(sorted(results, key=lambda item: item.hypothesis_id)),
        n_primary=len(hierarchy.primary_by_id),
        n_primary_selected=len(selected_primary),
        secondary_test_level=child_level,
        release_status=release_status,
        release_reason_code=release_reason,
        calibration_gate_id=calibration_gate_id,
        calibration_evidence_id=calibration_evidence_id,
        calibration_protocol_id=calibration_protocol_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenHierarchicalFDRCollection:
    """Producer-owned exact-universe collection of hierarchical FDR results."""

    universe_id: str
    procedure_id: str
    alpha: float
    q_value_scope: str
    p_value_record_ids: tuple[str, ...]
    records: tuple[HierarchicalFDRHypothesisResult, ...]
    n_primary: int
    n_primary_selected: int
    secondary_test_level: float
    release_status: HierarchicalFDRReleaseStatus
    release_reason_code: str | None
    calibration_gate_id: str | None
    calibration_evidence_id: str | None
    calibration_protocol_id: str | None
    collection_id: str
    _universe: FrozenHypothesisUniverse
    _spec: FrozenHierarchicalFDRSpec
    _p_value_records: tuple[HypothesisPValueRecord, ...]
    _calibration_gate: G3FrequencyCalibrationGate | None
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "FrozenHierarchicalFDRCollection is producer-owned; use "
            "evaluate_hierarchical_fdr()"
        )

    @property
    def q_value_release_allowed(self) -> bool:
        return self.release_status is HierarchicalFDRReleaseStatus.RELEASED

    def _identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "procedure_id": self.procedure_id,
            "alpha": self.alpha,
            "q_value_scope": self.q_value_scope,
            "p_value_record_ids": list(self.p_value_record_ids),
            "result_ids": [item.result_id for item in self.records],
            "n_primary": self.n_primary,
            "n_primary_selected": self.n_primary_selected,
            "secondary_test_level": self.secondary_test_level,
            "release_status": self.release_status.value,
            "release_reason_code": self.release_reason_code,
            "calibration_gate_id": self.calibration_gate_id,
            "calibration_evidence_id": self.calibration_evidence_id,
            "calibration_protocol_id": self.calibration_protocol_id,
        }

    def _require_intact(self) -> None:
        try:
            expected = _evaluate(
                universe=self._universe,
                spec=self._spec,
                p_value_records=self._p_value_records,
                calibration_gate=self._calibration_gate,
            )
            for record in self.records:
                record._require_intact()
            expected_id = stable_id(
                "frozen_hierarchical_fdr_collection",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _RESULT_MARKER
                and self.universe_id == self._universe.universe_id
                and self.procedure_id == self._spec.procedure_id
                and self.alpha == self._spec.alpha
                and self.q_value_scope == self._spec.q_value_scope
                and self.p_value_record_ids
                == tuple(item.record_id for item in self._p_value_records)
                and tuple(item.result_id for item in self.records)
                == tuple(item.result_id for item in expected.records)
                and self.n_primary == expected.n_primary
                and self.n_primary_selected == expected.n_primary_selected
                and self.secondary_test_level == expected.secondary_test_level
                and self.release_status is expected.release_status
                and self.release_reason_code == expected.release_reason_code
                and self.calibration_gate_id == expected.calibration_gate_id
                and self.calibration_evidence_id == expected.calibration_evidence_id
                and self.calibration_protocol_id == expected.calibration_protocol_id
                and self.collection_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Hierarchical FDR collection failed integrity validation",
                code="hierarchical_fdr_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the exact frozen universe and p-values",
            ) from error
        if not valid:
            raise ContractError(
                "Hierarchical FDR collection failed integrity validation",
                code="hierarchical_fdr_collection_integrity_violation",
                field="collection_id",
                remediation="Recompute from the exact frozen universe and p-values",
            )

    def result_for(self, hypothesis_id: str) -> HierarchicalFDRHypothesisResult:
        """Return one exact result or fail on an unknown hypothesis ID."""

        self._require_intact()
        identifier = _name(hypothesis_id, field_name="hypothesis_id")
        matched = tuple(
            item for item in self.records if item.hypothesis_id == identifier
        )
        if len(matched) != 1:
            raise ContractError(
                "Hypothesis is absent from the hierarchical FDR collection",
                code="hierarchical_fdr_result_not_found",
                field="hypothesis_id",
                remediation="Use an exact hypothesis ID from the frozen universe",
            )
        return matched[0]

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "collection_id": self.collection_id,
            **self._identity_payload(),
            "q_value_release_allowed": self.q_value_release_allowed,
            "records": [item.to_dict() for item in self.records],
        }


def evaluate_hierarchical_fdr(
    universe: FrozenHypothesisUniverse,
    p_value_records: Sequence[HypothesisPValueRecord],
    *,
    spec: FrozenHierarchicalFDRSpec,
    calibration_gate: G3FrequencyCalibrationGate | None = None,
) -> FrozenHierarchicalFDRCollection:
    """Compute exact two-level candidate q-values over one frozen universe."""

    if not isinstance(spec, FrozenHierarchicalFDRSpec):
        raise TypeError("spec must be a FrozenHierarchicalFDRSpec")
    gate = _validated_calibration_gate(calibration_gate)
    ordered = _validated_records(universe, p_value_records)
    evaluation = _evaluate(
        universe=universe,
        spec=spec,
        p_value_records=ordered,
        calibration_gate=gate,
    )
    self = object.__new__(FrozenHierarchicalFDRCollection)
    values: dict[str, object] = {
        "universe_id": universe.universe_id,
        "procedure_id": spec.procedure_id,
        "alpha": spec.alpha,
        "q_value_scope": spec.q_value_scope,
        "p_value_record_ids": tuple(item.record_id for item in ordered),
        "records": evaluation.records,
        "n_primary": evaluation.n_primary,
        "n_primary_selected": evaluation.n_primary_selected,
        "secondary_test_level": evaluation.secondary_test_level,
        "release_status": evaluation.release_status,
        "release_reason_code": evaluation.release_reason_code,
        "calibration_gate_id": evaluation.calibration_gate_id,
        "calibration_evidence_id": evaluation.calibration_evidence_id,
        "calibration_protocol_id": evaluation.calibration_protocol_id,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_universe", universe)
    object.__setattr__(self, "_spec", spec)
    object.__setattr__(self, "_p_value_records", ordered)
    object.__setattr__(self, "_calibration_gate", gate)
    object.__setattr__(self, "_producer_marker", _RESULT_MARKER)
    object.__setattr__(
        self,
        "collection_id",
        stable_id(
            "frozen_hierarchical_fdr_collection",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    return self


__all__ = [
    "FrozenHierarchicalFDRCollection",
    "FrozenHierarchicalFDRSpec",
    "HierarchicalFDRHypothesisResult",
    "HierarchicalFDRReleaseStatus",
    "HypothesisPValueRecord",
    "evaluate_hierarchical_fdr",
    "freeze_hierarchical_fdr_spec",
]
