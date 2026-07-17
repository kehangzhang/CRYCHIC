"""Unbound selection frequency over complete subject-bootstrap opportunities.

This low-level numeric primitive is not a workflow adapter. A later adapter must
bind intact V03-11 resampled-attribution collections before public release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import fsum
from typing import cast

from crychic.core import ContractError, stable_id

_SCHEMA_VERSION = "1.0.0"
_SUBJECT_BOOTSTRAP = "subject_bootstrap"
_MINIMUM_BOOTSTRAPS = 1_000
_OPPORTUNITY_GRAIN = "equal_bootstrap_mean_of_outer_fold_selection_v1"
_FREQUENCY_SEMANTICS = (
    "complete_subject_bootstrap_equal_plan_mean_of_outer_fold_selection_v1"
)
_SOURCE_BINDING_STATUS = "unbound_numeric_primitive_requires_frozen_workflow_adapter"
_PRODUCER_MARKER = "crychic.inference.selection_frequency.v1"


def _canonical_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _canonical_fold_id(value: object) -> str | None:
    if value is None:
        return None
    return _canonical_name(value, field_name="fold_id")


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


class SelectionFrequencyStatus(StrEnum):
    """Availability of a selection opportunity or its bootstrap summary."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapSelectionOpportunity:
    """One pre-registered plan/resample/fold opportunity key."""

    plan_id: str
    resample_index: int
    fold_id: str | None
    opportunity_id: str = field(init=False)

    def __post_init__(self) -> None:
        plan_id = _canonical_name(self.plan_id, field_name="plan_id")
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be an integer >= 0")
        fold_id = _canonical_fold_id(self.fold_id)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(
            self,
            "opportunity_id",
            stable_id(
                "bootstrap_selection_opportunity",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "resample_index": self.resample_index,
            "fold_id": self.fold_id,
            "opportunity_grain": _OPPORTUNITY_GRAIN,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            repeated = BootstrapSelectionOpportunity(
                plan_id=self.plan_id,
                resample_index=self.resample_index,
                fold_id=self.fold_id,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.opportunity_id == repeated.opportunity_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Bootstrap selection opportunity failed integrity validation",
                code="bootstrap_selection_opportunity_integrity_violation",
                field="opportunity_id",
                remediation="Recreate the opportunity from frozen plan/fold lineage",
            ) from error
        if not valid:
            raise ContractError(
                "Bootstrap selection opportunity failed integrity validation",
                code="bootstrap_selection_opportunity_integrity_violation",
                field="opportunity_id",
                remediation="Recreate the opportunity from frozen plan/fold lineage",
            )

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def public_release_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"opportunity_id": self.opportunity_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapSelectionRecord:
    """One typed outer-fold selection opportunity in one bootstrap plan."""

    plan_id: str
    resample_index: int
    fold_id: str | None
    target_id: str
    hypothesis_universe_id: str
    hypothesis_id: str
    crossfit_spec_id: str
    selection_rule_id: str
    score_version: str
    selected: bool | None
    status: SelectionFrequencyStatus
    reason_code: str | None
    resampling_kind: str = _SUBJECT_BOOTSTRAP
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        plan_id = _canonical_name(self.plan_id, field_name="plan_id")
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be an integer >= 0")
        fold_id = _canonical_fold_id(self.fold_id)
        target_id = _canonical_name(self.target_id, field_name="target_id")
        universe_id = _canonical_name(
            self.hypothesis_universe_id,
            field_name="hypothesis_universe_id",
        )
        hypothesis_id = _canonical_name(
            self.hypothesis_id,
            field_name="hypothesis_id",
        )
        crossfit_spec_id = _canonical_name(
            self.crossfit_spec_id,
            field_name="crossfit_spec_id",
        )
        selection_rule_id = _canonical_name(
            self.selection_rule_id,
            field_name="selection_rule_id",
        )
        score_version = _canonical_name(
            self.score_version,
            field_name="score_version",
        )
        resampling_kind = _canonical_name(
            self.resampling_kind,
            field_name="resampling_kind",
        )
        if resampling_kind != _SUBJECT_BOOTSTRAP:
            raise ValueError("BootstrapSelectionRecord accepts subject_bootstrap only")
        status = SelectionFrequencyStatus(self.status)
        if status is SelectionFrequencyStatus.OBSERVED:
            if fold_id is None:
                raise ValueError("observed selection records require fold_id")
            if not isinstance(self.selected, bool):
                raise ValueError("observed selection records require bool selected")
            if self.reason_code is not None:
                raise ValueError("observed selection records cannot have reason_code")
            reason_code = None
        else:
            if self.selected is not None:
                raise ValueError("non-observed selection records require selected=None")
            reason_code = _canonical_name(
                self.reason_code,
                field_name="reason_code",
            )
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "hypothesis_universe_id", universe_id)
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "crossfit_spec_id", crossfit_spec_id)
        object.__setattr__(self, "selection_rule_id", selection_rule_id)
        object.__setattr__(self, "score_version", score_version)
        object.__setattr__(self, "resampling_kind", resampling_kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(
            self,
            "record_id",
            stable_id(
                "bootstrap_selection_record",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "resample_index": self.resample_index,
            "fold_id": self.fold_id,
            "target_id": self.target_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hypothesis_id": self.hypothesis_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "selection_rule_id": self.selection_rule_id,
            "score_version": self.score_version,
            "selected": self.selected,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "resampling_kind": self.resampling_kind,
            "opportunity_grain": _OPPORTUNITY_GRAIN,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            repeated = BootstrapSelectionRecord(
                plan_id=self.plan_id,
                resample_index=self.resample_index,
                fold_id=self.fold_id,
                target_id=self.target_id,
                hypothesis_universe_id=self.hypothesis_universe_id,
                hypothesis_id=self.hypothesis_id,
                crossfit_spec_id=self.crossfit_spec_id,
                selection_rule_id=self.selection_rule_id,
                score_version=self.score_version,
                selected=self.selected,
                status=self.status,
                reason_code=self.reason_code,
                resampling_kind=self.resampling_kind,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.record_id == repeated.record_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Bootstrap selection record failed integrity validation",
                code="bootstrap_selection_record_integrity_violation",
                field="record_id",
                remediation="Recreate the record from a frozen source event",
            ) from error
        if not valid:
            raise ContractError(
                "Bootstrap selection record failed integrity validation",
                code="bootstrap_selection_record_integrity_violation",
                field="record_id",
                remediation="Recreate the record from a frozen source event",
            )

    @property
    def opportunity_grain(self) -> str:
        return _OPPORTUNITY_GRAIN

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def public_release_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"record_id": self.record_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionFrequencySpec:
    """Freeze one target, selection rule, and exact bootstrap plan set."""

    target_id: str
    hypothesis_universe_id: str
    hypothesis_id: str
    crossfit_spec_id: str
    selection_rule_id: str
    score_version: str
    subject_bootstrap_plan_ids: tuple[str, ...]
    expected_opportunities: tuple[BootstrapSelectionOpportunity, ...]
    minimum_bootstraps: int = _MINIMUM_BOOTSTRAPS
    opportunity_grain: str = _OPPORTUNITY_GRAIN
    bootstrap_plan_set_id: str = field(init=False)
    opportunity_manifest_id: str = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        target_id = _canonical_name(self.target_id, field_name="target_id")
        universe_id = _canonical_name(
            self.hypothesis_universe_id,
            field_name="hypothesis_universe_id",
        )
        hypothesis_id = _canonical_name(
            self.hypothesis_id,
            field_name="hypothesis_id",
        )
        crossfit_spec_id = _canonical_name(
            self.crossfit_spec_id,
            field_name="crossfit_spec_id",
        )
        selection_rule_id = _canonical_name(
            self.selection_rule_id,
            field_name="selection_rule_id",
        )
        score_version = _canonical_name(
            self.score_version,
            field_name="score_version",
        )
        plan_ids = tuple(
            sorted(
                _canonical_name(value, field_name="subject_bootstrap_plan_ids")
                for value in self.subject_bootstrap_plan_ids
            )
        )
        if not plan_ids:
            raise ValueError("subject_bootstrap_plan_ids cannot be empty")
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("subject_bootstrap_plan_ids must be unique")
        if not self.expected_opportunities:
            raise ValueError("expected_opportunities cannot be empty")
        if any(
            not isinstance(item, BootstrapSelectionOpportunity)
            for item in self.expected_opportunities
        ):
            raise TypeError(
                "expected_opportunities must contain "
                "BootstrapSelectionOpportunity values"
            )
        for item in self.expected_opportunities:
            item._require_intact()
        opportunities = tuple(
            sorted(
                self.expected_opportunities,
                key=lambda item: (
                    item.resample_index,
                    item.plan_id,
                    item.fold_id is None,
                    item.fold_id or "",
                ),
            )
        )
        opportunity_keys = tuple(
            (item.plan_id, item.resample_index, item.fold_id) for item in opportunities
        )
        if len(opportunity_keys) != len(set(opportunity_keys)) or len(
            {item.opportunity_id for item in opportunities}
        ) != len(opportunities):
            raise ValueError("expected_opportunities must be unique")
        opportunity_plan_ids = tuple(sorted({item.plan_id for item in opportunities}))
        if opportunity_plan_ids != plan_ids:
            raise ValueError(
                "expected_opportunities must exactly cover bootstrap plan IDs"
            )
        plan_to_indices: dict[str, set[int]] = {}
        index_to_plans: dict[int, set[str]] = {}
        plan_to_folds: dict[str, list[str | None]] = {}
        for item in opportunities:
            plan_to_indices.setdefault(item.plan_id, set()).add(item.resample_index)
            index_to_plans.setdefault(item.resample_index, set()).add(item.plan_id)
            plan_to_folds.setdefault(item.plan_id, []).append(item.fold_id)
        if any(len(values) != 1 for values in plan_to_indices.values()):
            raise ValueError(
                "each bootstrap plan must map to one expected resample index"
            )
        if any(len(values) != 1 for values in index_to_plans.values()):
            raise ValueError(
                "each expected resample index must map to one bootstrap plan"
            )
        if any(None in folds and len(folds) != 1 for folds in plan_to_folds.values()):
            raise ValueError(
                "a foldless expected opportunity must be the plan's only opportunity"
            )
        if (
            isinstance(self.minimum_bootstraps, bool)
            or not isinstance(self.minimum_bootstraps, int)
            or self.minimum_bootstraps != _MINIMUM_BOOTSTRAPS
        ):
            raise ValueError("minimum_bootstraps is fixed at 1000")
        opportunity_grain = _canonical_name(
            self.opportunity_grain,
            field_name="opportunity_grain",
        )
        if opportunity_grain != _OPPORTUNITY_GRAIN:
            raise ValueError(f"opportunity_grain is fixed at {_OPPORTUNITY_GRAIN}")
        plan_set_id = stable_id(
            "subject_bootstrap_selection_plan_set",
            {"plan_ids": list(plan_ids)},
            schema_version=_SCHEMA_VERSION,
        )
        opportunity_manifest_id = stable_id(
            "bootstrap_selection_opportunity_manifest",
            {"opportunity_ids": [item.opportunity_id for item in opportunities]},
            schema_version=_SCHEMA_VERSION,
        )
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "hypothesis_universe_id", universe_id)
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "crossfit_spec_id", crossfit_spec_id)
        object.__setattr__(self, "selection_rule_id", selection_rule_id)
        object.__setattr__(self, "score_version", score_version)
        object.__setattr__(self, "subject_bootstrap_plan_ids", plan_ids)
        object.__setattr__(self, "expected_opportunities", opportunities)
        object.__setattr__(self, "opportunity_grain", opportunity_grain)
        object.__setattr__(self, "bootstrap_plan_set_id", plan_set_id)
        object.__setattr__(
            self,
            "opportunity_manifest_id",
            opportunity_manifest_id,
        )
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "selection_frequency_spec",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hypothesis_id": self.hypothesis_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "selection_rule_id": self.selection_rule_id,
            "score_version": self.score_version,
            "bootstrap_plan_set_id": self.bootstrap_plan_set_id,
            "subject_bootstrap_plan_ids": list(self.subject_bootstrap_plan_ids),
            "opportunity_manifest_id": self.opportunity_manifest_id,
            "expected_opportunity_ids": [
                item.opportunity_id for item in self.expected_opportunities
            ],
            "minimum_bootstraps": self.minimum_bootstraps,
            "opportunity_grain": self.opportunity_grain,
            "selection_frequency_semantics": _FREQUENCY_SEMANTICS,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
        }

    def _require_intact(self) -> None:
        try:
            repeated = SelectionFrequencySpec(
                target_id=self.target_id,
                hypothesis_universe_id=self.hypothesis_universe_id,
                hypothesis_id=self.hypothesis_id,
                crossfit_spec_id=self.crossfit_spec_id,
                selection_rule_id=self.selection_rule_id,
                score_version=self.score_version,
                subject_bootstrap_plan_ids=self.subject_bootstrap_plan_ids,
                expected_opportunities=self.expected_opportunities,
                minimum_bootstraps=self.minimum_bootstraps,
                opportunity_grain=self.opportunity_grain,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.bootstrap_plan_set_id == repeated.bootstrap_plan_set_id
                and self.opportunity_manifest_id == repeated.opportunity_manifest_id
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Selection frequency spec failed integrity validation",
                code="selection_frequency_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the spec from the frozen plan universe",
            ) from error
        if not valid:
            raise ContractError(
                "Selection frequency spec failed integrity validation",
                code="selection_frequency_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the spec from the frozen plan universe",
            )

    @property
    def selection_frequency_semantics(self) -> str:
        return _FREQUENCY_SEMANTICS

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def public_release_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            **self._identity_payload(),
            "expected_opportunities": [
                item.to_dict() for item in self.expected_opportunities
            ],
        }


@dataclass(frozen=True, slots=True)
class _SelectionFrequencyEvaluation:
    records: tuple[BootstrapSelectionRecord, ...]
    n_bootstrap_plans_total: int
    n_bootstrap_plans_observed: int
    n_bootstrap_plans_not_estimable: int
    n_bootstrap_plans_failed: int
    n_event_rows_total: int
    n_observed_event_rows: int
    n_not_estimable_event_rows: int
    n_failed_event_rows: int
    n_selected_event_rows: int
    selection_frequency: float | None
    status: SelectionFrequencyStatus
    reason_code: str | None


def _evaluate_selection_frequency(
    spec: SelectionFrequencySpec,
    records: tuple[BootstrapSelectionRecord, ...],
) -> _SelectionFrequencyEvaluation:
    spec._require_intact()
    if not records:
        raise _contract_error(
            "Bootstrap selection records are missing",
            code="selection_frequency_plan_set_mismatch",
            field="subject_bootstrap_plan_ids",
            remediation="Supply every frozen bootstrap plan",
        )
    if any(not isinstance(item, BootstrapSelectionRecord) for item in records):
        raise TypeError("records must contain BootstrapSelectionRecord values")
    for item in records:
        item._require_intact()
    canonical = tuple(
        sorted(
            records,
            key=lambda item: (
                item.resample_index,
                item.plan_id,
                item.fold_id is None,
                item.fold_id or "",
            ),
        )
    )
    if any(item.resampling_kind != _SUBJECT_BOOTSTRAP for item in canonical):
        raise _contract_error(
            "Selection frequency accepts subject-bootstrap records only",
            code="selection_frequency_nonbootstrap_forbidden",
            field="resampling_kind",
            remediation="Remove non-bootstrap records",
        )
    observed_plan_ids = tuple(sorted({item.plan_id for item in canonical}))
    if observed_plan_ids != spec.subject_bootstrap_plan_ids:
        raise _contract_error(
            "Observed bootstrap plans do not match the frozen plan set",
            code="selection_frequency_plan_set_mismatch",
            field="subject_bootstrap_plan_ids",
            remediation="Supply the exact frozen bootstrap plan set",
        )
    if len({item.record_id for item in canonical}) != len(canonical):
        raise _contract_error(
            "Bootstrap selection records contain duplicate identities",
            code="selection_frequency_duplicate_record",
            field="record_id",
            remediation="Supply each source event exactly once",
        )
    plan_fold_keys = tuple((item.plan_id, item.fold_id) for item in canonical)
    if len(set(plan_fold_keys)) != len(plan_fold_keys):
        raise _contract_error(
            "Bootstrap selection records contain duplicate plan/fold events",
            code="selection_frequency_duplicate_plan_fold",
            field="plan_id,fold_id",
            remediation="Supply every outer-fold opportunity exactly once",
        )
    by_plan: dict[str, list[BootstrapSelectionRecord]] = {
        plan_id: [] for plan_id in spec.subject_bootstrap_plan_ids
    }
    plan_to_index: dict[str, int] = {}
    index_to_plan: dict[int, str] = {}
    for item in canonical:
        previous_index = plan_to_index.setdefault(item.plan_id, item.resample_index)
        if previous_index != item.resample_index:
            raise _contract_error(
                "One bootstrap plan maps to multiple resample indices",
                code="selection_frequency_plan_resample_mismatch",
                field="plan_id,resample_index",
                remediation="Use one frozen resample index for every plan",
            )
        previous_plan = index_to_plan.setdefault(item.resample_index, item.plan_id)
        if previous_plan != item.plan_id:
            raise _contract_error(
                "One resample index maps to multiple bootstrap plans",
                code="selection_frequency_duplicate_resample_index",
                field="plan_id,resample_index",
                remediation="Use unique bootstrap plan/resample provenance",
            )
        by_plan[item.plan_id].append(item)
    for plan_records in by_plan.values():
        if (
            any(item.fold_id is None for item in plan_records)
            and len(plan_records) != 1
        ):
            raise _contract_error(
                "A foldless plan event must be the plan's only event",
                code="selection_frequency_foldless_plan_ambiguous",
                field="plan_id,fold_id",
                remediation="Use one typed foldless failure/NE or unique fold events",
            )
    actual_opportunity_keys = tuple(
        (item.plan_id, item.resample_index, item.fold_id) for item in canonical
    )
    expected_opportunity_keys = tuple(
        (item.plan_id, item.resample_index, item.fold_id)
        for item in spec.expected_opportunities
    )
    if actual_opportunity_keys != expected_opportunity_keys:
        raise _contract_error(
            "Selection records do not exactly cover pre-registered opportunities",
            code="selection_frequency_opportunity_coverage_mismatch",
            field="plan_id,resample_index,fold_id",
            remediation="Supply every expected opportunity exactly once and no extras",
        )
    lineage = (
        spec.target_id,
        spec.hypothesis_universe_id,
        spec.hypothesis_id,
        spec.crossfit_spec_id,
        spec.selection_rule_id,
        spec.score_version,
    )
    if any(
        (
            item.target_id,
            item.hypothesis_universe_id,
            item.hypothesis_id,
            item.crossfit_spec_id,
            item.selection_rule_id,
            item.score_version,
        )
        != lineage
        for item in canonical
    ):
        raise _contract_error(
            "Bootstrap selection record lineage does not match the frozen spec",
            code="selection_frequency_record_lineage_mismatch",
            field=(
                "target_id,hypothesis_universe_id,hypothesis_id,crossfit_spec_id,"
                "selection_rule_id,score_version"
            ),
            remediation="Use events from only the exact frozen target and rule",
        )
    n_observed = sum(
        item.status is SelectionFrequencyStatus.OBSERVED for item in canonical
    )
    n_not_estimable = sum(
        item.status is SelectionFrequencyStatus.NOT_ESTIMABLE for item in canonical
    )
    n_failed = sum(item.status is SelectionFrequencyStatus.FAILED for item in canonical)
    n_selected = sum(
        item.status is SelectionFrequencyStatus.OBSERVED and item.selected is True
        for item in canonical
    )
    plan_statuses: dict[str, SelectionFrequencyStatus] = {}
    for plan_id, plan_records in by_plan.items():
        statuses = {item.status for item in plan_records}
        if SelectionFrequencyStatus.FAILED in statuses:
            plan_statuses[plan_id] = SelectionFrequencyStatus.FAILED
        elif SelectionFrequencyStatus.NOT_ESTIMABLE in statuses:
            plan_statuses[plan_id] = SelectionFrequencyStatus.NOT_ESTIMABLE
        else:
            plan_statuses[plan_id] = SelectionFrequencyStatus.OBSERVED
    n_plan_failed = sum(
        status is SelectionFrequencyStatus.FAILED for status in plan_statuses.values()
    )
    n_plan_not_estimable = sum(
        status is SelectionFrequencyStatus.NOT_ESTIMABLE
        for status in plan_statuses.values()
    )
    n_plan_observed = sum(
        status is SelectionFrequencyStatus.OBSERVED for status in plan_statuses.values()
    )
    frequency: float | None = None
    reason_code: str | None
    status: SelectionFrequencyStatus
    if n_failed:
        status = SelectionFrequencyStatus.FAILED
        reason_code = "selection_frequency_bootstrap_event_failed"
    elif n_not_estimable:
        status = SelectionFrequencyStatus.NOT_ESTIMABLE
        reason_code = "selection_frequency_bootstrap_event_not_estimable"
    elif len(by_plan) < spec.minimum_bootstraps:
        status = SelectionFrequencyStatus.NOT_ESTIMABLE
        reason_code = "selection_frequency_bootstrap_count_below_1000"
    else:
        plan_means = tuple(
            fsum(1.0 if item.selected is True else 0.0 for item in plan_records)
            / len(plan_records)
            for plan_records in by_plan.values()
        )
        frequency = fsum(plan_means) / len(plan_means)
        status = SelectionFrequencyStatus.OBSERVED
        reason_code = None
    return _SelectionFrequencyEvaluation(
        records=canonical,
        n_bootstrap_plans_total=len(by_plan),
        n_bootstrap_plans_observed=n_plan_observed,
        n_bootstrap_plans_not_estimable=n_plan_not_estimable,
        n_bootstrap_plans_failed=n_plan_failed,
        n_event_rows_total=len(canonical),
        n_observed_event_rows=n_observed,
        n_not_estimable_event_rows=n_not_estimable,
        n_failed_event_rows=n_failed,
        n_selected_event_rows=n_selected,
        selection_frequency=frequency,
        status=status,
        reason_code=reason_code,
    )


@dataclass(frozen=True, slots=True, init=False)
class SelectionFrequencyResult:
    """Producer-owned equal-bootstrap selection frequency diagnostic."""

    selection_frequency_spec_id: str
    target_id: str
    hypothesis_universe_id: str
    hypothesis_id: str
    crossfit_spec_id: str
    selection_rule_id: str
    score_version: str
    bootstrap_plan_set_id: str
    opportunity_manifest_id: str
    subject_bootstrap_plan_ids: tuple[str, ...]
    expected_opportunity_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    minimum_bootstraps: int
    opportunity_grain: str
    n_bootstrap_plans_total: int
    n_bootstrap_plans_observed: int
    n_bootstrap_plans_not_estimable: int
    n_bootstrap_plans_failed: int
    n_event_rows_total: int
    n_observed_event_rows: int
    n_not_estimable_event_rows: int
    n_failed_event_rows: int
    n_selected_event_rows: int
    selection_frequency: float | None
    status: SelectionFrequencyStatus
    reason_code: str | None
    result_id: str
    _spec: SelectionFrequencySpec
    _source_records: tuple[BootstrapSelectionRecord, ...]
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "SelectionFrequencyResult is producer-owned; use "
            "summarize_selection_frequency()"
        )

    @property
    def selection_frequency_semantics(self) -> str:
        return _FREQUENCY_SEMANTICS

    @property
    def source_binding_status(self) -> str:
        return _SOURCE_BINDING_STATUS

    @property
    def public_release_allowed(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "selection_frequency_spec_id": self.selection_frequency_spec_id,
            "target_id": self.target_id,
            "hypothesis_universe_id": self.hypothesis_universe_id,
            "hypothesis_id": self.hypothesis_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "selection_rule_id": self.selection_rule_id,
            "score_version": self.score_version,
            "bootstrap_plan_set_id": self.bootstrap_plan_set_id,
            "opportunity_manifest_id": self.opportunity_manifest_id,
            "subject_bootstrap_plan_ids": list(self.subject_bootstrap_plan_ids),
            "expected_opportunity_ids": list(self.expected_opportunity_ids),
            "source_record_ids": list(self.source_record_ids),
            "minimum_bootstraps": self.minimum_bootstraps,
            "opportunity_grain": self.opportunity_grain,
            "n_bootstrap_plans_total": self.n_bootstrap_plans_total,
            "n_bootstrap_plans_observed": self.n_bootstrap_plans_observed,
            "n_bootstrap_plans_not_estimable": (self.n_bootstrap_plans_not_estimable),
            "n_bootstrap_plans_failed": self.n_bootstrap_plans_failed,
            "n_event_rows_total": self.n_event_rows_total,
            "n_observed_event_rows": self.n_observed_event_rows,
            "n_not_estimable_event_rows": self.n_not_estimable_event_rows,
            "n_failed_event_rows": self.n_failed_event_rows,
            "n_selected_event_rows": self.n_selected_event_rows,
            "selection_frequency": self.selection_frequency,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "selection_frequency_semantics": _FREQUENCY_SEMANTICS,
            "source_binding_status": _SOURCE_BINDING_STATUS,
            "public_release_allowed": False,
            "producer_marker": _PRODUCER_MARKER,
        }

    def _require_intact(self) -> None:
        try:
            evaluation = _evaluate_selection_frequency(
                self._spec,
                self._source_records,
            )
            expected_values = _result_values(self._spec, evaluation)
            expected_payload = _result_identity_payload(expected_values)
            expected_id = stable_id(
                "selection_frequency_result",
                expected_payload,
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self._identity_payload() == expected_payload
                and self.result_id == expected_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Selection frequency result failed integrity validation",
                code="selection_frequency_result_integrity_violation",
                field="result_id",
                remediation="Re-summarize the intact spec and source records",
            ) from error
        if not valid:
            raise ContractError(
                "Selection frequency result failed integrity validation",
                code="selection_frequency_result_integrity_violation",
                field="result_id",
                remediation="Re-summarize the intact spec and source records",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        payload = self._identity_payload()
        payload.pop("producer_marker")
        return {
            "result_id": self.result_id,
            **payload,
            "expected_opportunities": [
                item.to_dict() for item in self._spec.expected_opportunities
            ],
        }


def _result_values(
    spec: SelectionFrequencySpec,
    evaluation: _SelectionFrequencyEvaluation,
) -> dict[str, object]:
    return {
        "selection_frequency_spec_id": spec.spec_id,
        "target_id": spec.target_id,
        "hypothesis_universe_id": spec.hypothesis_universe_id,
        "hypothesis_id": spec.hypothesis_id,
        "crossfit_spec_id": spec.crossfit_spec_id,
        "selection_rule_id": spec.selection_rule_id,
        "score_version": spec.score_version,
        "bootstrap_plan_set_id": spec.bootstrap_plan_set_id,
        "opportunity_manifest_id": spec.opportunity_manifest_id,
        "subject_bootstrap_plan_ids": spec.subject_bootstrap_plan_ids,
        "expected_opportunity_ids": tuple(
            item.opportunity_id for item in spec.expected_opportunities
        ),
        "source_record_ids": tuple(item.record_id for item in evaluation.records),
        "minimum_bootstraps": spec.minimum_bootstraps,
        "opportunity_grain": spec.opportunity_grain,
        "n_bootstrap_plans_total": evaluation.n_bootstrap_plans_total,
        "n_bootstrap_plans_observed": evaluation.n_bootstrap_plans_observed,
        "n_bootstrap_plans_not_estimable": (evaluation.n_bootstrap_plans_not_estimable),
        "n_bootstrap_plans_failed": evaluation.n_bootstrap_plans_failed,
        "n_event_rows_total": evaluation.n_event_rows_total,
        "n_observed_event_rows": evaluation.n_observed_event_rows,
        "n_not_estimable_event_rows": evaluation.n_not_estimable_event_rows,
        "n_failed_event_rows": evaluation.n_failed_event_rows,
        "n_selected_event_rows": evaluation.n_selected_event_rows,
        "selection_frequency": evaluation.selection_frequency,
        "status": evaluation.status,
        "reason_code": evaluation.reason_code,
    }


def _result_identity_payload(values: dict[str, object]) -> dict[str, object]:
    status = cast(SelectionFrequencyStatus, values["status"])
    return {
        **values,
        "subject_bootstrap_plan_ids": list(
            cast(tuple[str, ...], values["subject_bootstrap_plan_ids"])
        ),
        "expected_opportunity_ids": list(
            cast(tuple[str, ...], values["expected_opportunity_ids"])
        ),
        "source_record_ids": list(cast(tuple[str, ...], values["source_record_ids"])),
        "status": status.value,
        "selection_frequency_semantics": _FREQUENCY_SEMANTICS,
        "source_binding_status": _SOURCE_BINDING_STATUS,
        "public_release_allowed": False,
        "producer_marker": _PRODUCER_MARKER,
    }


def summarize_selection_frequency(
    spec: SelectionFrequencySpec,
    records: tuple[BootstrapSelectionRecord, ...],
) -> SelectionFrequencyResult:
    """Summarize complete bootstrap opportunities with equal plan weight."""

    if not isinstance(spec, SelectionFrequencySpec):
        raise TypeError("spec must be SelectionFrequencySpec")
    supplied = tuple(records)
    evaluation = _evaluate_selection_frequency(spec, supplied)
    values = _result_values(spec, evaluation)
    self = object.__new__(SelectionFrequencyResult)
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(self, "_spec", spec)
    object.__setattr__(self, "_source_records", evaluation.records)
    object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
    object.__setattr__(
        self,
        "result_id",
        stable_id(
            "selection_frequency_result",
            self._identity_payload(),
            schema_version=_SCHEMA_VERSION,
        ),
    )
    self._require_intact()
    return self


__all__ = [
    "BootstrapSelectionOpportunity",
    "BootstrapSelectionRecord",
    "SelectionFrequencyResult",
    "SelectionFrequencySpec",
    "SelectionFrequencyStatus",
    "summarize_selection_frequency",
]
