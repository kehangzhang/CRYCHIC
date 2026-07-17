"""Degree- and evidence-matched active-edge null resource plans.

This module only rewires an immutable :class:`~crychic.resources.TargetPrior`.
It does not inspect outcomes, fit models, or invoke workflow code.  A workflow
must pass the training-fold, context-label-blind prior scope and rerun the full
pipeline after materializing an observed plan.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum

from crychic.core import ContractError, SeedLineage, stable_id
from crychic.resources import MappingReport, TargetPrior

_RANK_TIER_UPPER_BOUNDS = (10, 25, 50, 100, 250)
_SWITCHES_PER_EDGE = 20
_MAX_OVERLAP_FRACTION = 0.05
_MAX_ATTEMPTS_PER_REQUIRED_SWITCH = 100
_SPEC_SCHEMA_VERSION = "1"
_PLAN_SCHEMA_VERSION = "1"
_ACTIVE_NULL_PLAN_PRODUCER_TOKEN = object()


class ActiveNullPlanStatus(StrEnum):
    """Whether a target-prior reassignment is usable."""

    OBSERVED = "observed"
    NOT_ESTIMABLE = "not_estimable"


class ActiveNullReasonCode(StrEnum):
    """Stable fail-closed reasons owned by the active-null planner."""

    REASSIGNMENT_INFEASIBLE = "active_null_reassignment_infeasible"
    INSUFFICIENT_REWIRING = "active_null_insufficient_rewiring"


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveNullSpec:
    """Frozen ADR-013 double-edge-switch contract."""

    rank_tier_upper_bounds: tuple[int, ...] = _RANK_TIER_UPPER_BOUNDS
    accepted_switches_per_edge: int = _SWITCHES_PER_EDGE
    max_overlap_fraction: float = _MAX_OVERLAP_FRACTION
    max_attempts_per_required_switch: int = _MAX_ATTEMPTS_PER_REQUIRED_SWITCH
    active_null_spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.rank_tier_upper_bounds != _RANK_TIER_UPPER_BOUNDS:
            raise ValueError("ADR-013 rank tiers are frozen and cannot be changed")
        if self.accepted_switches_per_edge != _SWITCHES_PER_EDGE:
            raise ValueError("ADR-013 requires exactly 20 accepted switches per edge")
        if (
            not math.isfinite(self.max_overlap_fraction)
            or self.max_overlap_fraction != _MAX_OVERLAP_FRACTION
        ):
            raise ValueError("ADR-013 maximum source overlap is frozen at 0.05")
        if (
            isinstance(self.max_attempts_per_required_switch, bool)
            or not isinstance(self.max_attempts_per_required_switch, int)
            or self.max_attempts_per_required_switch
            != _MAX_ATTEMPTS_PER_REQUIRED_SWITCH
        ):
            raise ValueError("ADR-013 switch-attempt budget is frozen at 100")
        object.__setattr__(
            self,
            "active_null_spec_id",
            stable_id(
                "active_null_spec",
                self._identity_payload(),
                schema_version=_SPEC_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "rank_tier_upper_bounds": list(self.rank_tier_upper_bounds),
            "accepted_switches_per_edge": self.accepted_switches_per_edge,
            "max_overlap_fraction": self.max_overlap_fraction,
            "max_attempts_per_required_switch": (self.max_attempts_per_required_switch),
            "operation": "bipartite_double_edge_switch_within_rank_tier_v1",
            "weight_and_rank_attachment": "driver_stub",
        }

    def to_dict(self) -> dict[str, object]:
        """Return the immutable persisted representation."""

        return {
            "active_null_spec_id": self.active_null_spec_id,
            **self._identity_payload(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveNullPriorLink:
    """One reassigned link whose evidence remains attached to its driver stub."""

    driver_id: str
    target_id: str
    weight: float
    rank: int | None
    evidence_rank: int
    evidence_tier: int

    def __post_init__(self) -> None:
        for field_name in ("driver_id", "target_id"):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be a canonical non-empty string")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("active-null link weight must be finite and non-negative")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("active-null link rank must be a positive integer or None")
        if (
            isinstance(self.evidence_rank, bool)
            or not isinstance(self.evidence_rank, int)
            or self.evidence_rank < 1
        ):
            raise ValueError("evidence_rank must be a positive integer")
        if (
            isinstance(self.evidence_tier, bool)
            or not isinstance(self.evidence_tier, int)
            or self.evidence_tier < 0
        ):
            raise ValueError("evidence_tier must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """Return a canonical identity payload."""

        return {
            "driver_id": self.driver_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "rank": self.rank,
            "evidence_rank": self.evidence_rank,
            "evidence_tier": self.evidence_tier,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveNullTierDiagnostic:
    """Switch and overlap accounting for one non-empty evidence tier."""

    evidence_tier: int
    edge_count: int
    required_switch_count: int
    accepted_switch_count: int
    attempted_switch_count: int
    source_overlap_count: int | None

    def __post_init__(self) -> None:
        values = (
            self.evidence_tier,
            self.edge_count,
            self.required_switch_count,
            self.accepted_switch_count,
            self.attempted_switch_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("active-null tier diagnostic counts must be integers")
        if self.evidence_tier < 0 or self.edge_count < 1:
            raise ValueError("active-null tier and edge counts are invalid")
        if self.required_switch_count != _SWITCHES_PER_EDGE * self.edge_count:
            raise ValueError("tier required switches must equal 20 times edge count")
        if not 0 <= self.accepted_switch_count <= self.attempted_switch_count:
            raise ValueError("accepted switch count cannot exceed attempted switches")
        if self.source_overlap_count is not None and not (
            0 <= self.source_overlap_count <= self.edge_count
        ):
            raise ValueError("tier source overlap count is invalid")

    def to_dict(self) -> dict[str, object]:
        """Return a canonical identity payload."""

        return {
            "evidence_tier": self.evidence_tier,
            "edge_count": self.edge_count,
            "required_switch_count": self.required_switch_count,
            "accepted_switch_count": self.accepted_switch_count,
            "attempted_switch_count": self.attempted_switch_count,
            "source_overlap_count": self.source_overlap_count,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveNullPlan:
    """Immutable reassignment plan or typed not-estimable diagnostic."""

    active_null_spec_id: str
    source_prior_content_id: str
    seed_lineage: SeedLineage
    status: ActiveNullPlanStatus
    reason_code: ActiveNullReasonCode | None
    source_edge_count: int
    accepted_switch_count: int
    attempted_switch_count: int
    source_overlap_count: int | None
    source_overlap_fraction: float | None
    tier_diagnostics: tuple[ActiveNullTierDiagnostic, ...]
    reassigned_links: tuple[ActiveNullPriorLink, ...]
    _producer_token: object = field(repr=False, compare=False)
    active_null_plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self._producer_token is not _ACTIVE_NULL_PLAN_PRODUCER_TOKEN:
            raise ContractError(
                "Active-null plans must be created by the reviewed planner",
                code="caller_owned_active_null_plan_forbidden",
                field="active_null_plan",
                remediation="Call plan_active_null_target_prior",
            )
        status = ActiveNullPlanStatus(self.status)
        reason = (
            None if self.reason_code is None else ActiveNullReasonCode(self.reason_code)
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason)
        if not self.active_null_spec_id or not self.source_prior_content_id:
            raise ValueError("active-null source and spec identities cannot be empty")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        counts = (
            self.source_edge_count,
            self.accepted_switch_count,
            self.attempted_switch_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("active-null plan counts must be non-negative integers")
        if self.source_edge_count < 1:
            raise ValueError("active-null plan must bind at least one source edge")
        diagnostics = tuple(self.tier_diagnostics)
        if tuple(sorted(item.evidence_tier for item in diagnostics)) != tuple(
            item.evidence_tier for item in diagnostics
        ) or len({item.evidence_tier for item in diagnostics}) != len(diagnostics):
            raise ValueError("active-null tier diagnostics must be unique and sorted")
        links = tuple(
            sorted(
                self.reassigned_links,
                key=lambda link: (
                    link.driver_id,
                    link.evidence_rank,
                    link.target_id,
                    link.weight,
                ),
            )
        )
        object.__setattr__(self, "tier_diagnostics", diagnostics)
        object.__setattr__(self, "reassigned_links", links)
        if self.accepted_switch_count != sum(
            item.accepted_switch_count for item in diagnostics
        ) or self.attempted_switch_count != sum(
            item.attempted_switch_count for item in diagnostics
        ):
            raise ValueError("active-null plan totals do not match tier diagnostics")
        if status is ActiveNullPlanStatus.OBSERVED:
            if reason is not None:
                raise ValueError("observed active-null plan cannot have a reason code")
            if len(links) != self.source_edge_count:
                raise ValueError(
                    "observed active-null plan must retain every source edge"
                )
            if len({(link.driver_id, link.target_id) for link in links}) != len(links):
                raise ValueError("observed active-null plan contains duplicate links")
            if (
                self.source_overlap_count is None
                or self.source_overlap_fraction is None
            ):
                raise ValueError(
                    "observed active-null plan requires overlap diagnostics"
                )
            if not math.isclose(
                self.source_overlap_fraction,
                self.source_overlap_count / self.source_edge_count,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("active-null overlap count and fraction disagree")
            if self.source_overlap_fraction > _MAX_OVERLAP_FRACTION:
                raise ValueError(
                    "observed active-null plan exceeds source overlap limit"
                )
            if any(
                item.accepted_switch_count < item.required_switch_count
                for item in diagnostics
            ):
                raise ValueError("observed active-null plan has an under-switched tier")
            if sum(item.edge_count for item in diagnostics) != self.source_edge_count:
                raise ValueError("observed active-null tiers do not cover source edges")
        else:
            if reason is None:
                raise ValueError(
                    "not-estimable active-null plan requires a reason code"
                )
            if links:
                raise ValueError("not-estimable active-null plan cannot expose links")
            if (self.source_overlap_count is None) != (
                self.source_overlap_fraction is None
            ):
                raise ValueError("not-estimable overlap diagnostics must be paired")
            if self.source_overlap_count is not None:
                overlap_fraction = self.source_overlap_fraction
                if overlap_fraction is None or (
                    not 0 <= self.source_overlap_count <= self.source_edge_count
                    or not math.isclose(
                        overlap_fraction,
                        self.source_overlap_count / self.source_edge_count,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ):
                    raise ValueError("not-estimable overlap diagnostics disagree")
        object.__setattr__(
            self,
            "active_null_plan_id",
            stable_id(
                "active_null_plan",
                self._identity_components(),
                schema_version=_PLAN_SCHEMA_VERSION,
            ),
        )

    def _identity_components(self) -> dict[str, object]:
        """Return canonical-ID components without duplicating every link dict."""

        return {
            "active_null_spec_id": self.active_null_spec_id,
            "source_prior_content_id": self.source_prior_content_id,
            "seed_lineage": self.seed_lineage.to_dict(),
            "status": self.status.value,
            "reason_code": None if self.reason_code is None else self.reason_code.value,
            "source_edge_count": self.source_edge_count,
            "accepted_switch_count": self.accepted_switch_count,
            "attempted_switch_count": self.attempted_switch_count,
            "source_overlap_count": self.source_overlap_count,
            "source_overlap_fraction": self.source_overlap_fraction,
            "tier_diagnostics": self.tier_diagnostics,
            "reassigned_links": self.reassigned_links,
        }

    def _identity_payload(self) -> dict[str, object]:
        return {
            "active_null_spec_id": self.active_null_spec_id,
            "source_prior_content_id": self.source_prior_content_id,
            "seed_lineage": self.seed_lineage.to_dict(),
            "status": self.status.value,
            "reason_code": None if self.reason_code is None else self.reason_code.value,
            "source_edge_count": self.source_edge_count,
            "accepted_switch_count": self.accepted_switch_count,
            "attempted_switch_count": self.attempted_switch_count,
            "source_overlap_count": self.source_overlap_count,
            "source_overlap_fraction": self.source_overlap_fraction,
            "tier_diagnostics": [item.to_dict() for item in self.tier_diagnostics],
            "reassigned_links": [link.to_dict() for link in self.reassigned_links],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the immutable persisted representation."""

        return {
            "active_null_plan_id": self.active_null_plan_id,
            **self._identity_payload(),
        }

    def require_intact(self) -> None:
        """Reject post-construction mutation before the plan is consumed."""

        if self._producer_token is not _ACTIVE_NULL_PLAN_PRODUCER_TOKEN:
            raise ContractError(
                "Active-null plan producer binding is not intact",
                code="active_null_plan_integrity_violation",
                field="active_null_plan",
                remediation="Reject the artifact and regenerate the null plan",
            )
        expected = stable_id(
            "active_null_plan",
            self._identity_components(),
            schema_version=_PLAN_SCHEMA_VERSION,
        )
        if self.active_null_plan_id != expected:
            raise ContractError(
                "Active-null plan identity is not intact",
                code="active_null_plan_integrity_violation",
                field="active_null_plan_id",
                remediation="Reject the artifact and regenerate the null plan",
            )


def target_prior_content_id(prior: TargetPrior) -> str:
    """Return the content identity shared with training provenance."""

    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    # Canonical serialization supports dataclasses directly. Passing the intact
    # object emits the same JSON tree as ``asdict(prior)`` without first making
    # a second full copy of every sparse coordinate array.
    result: str = stable_id("target_prior_content", prior)
    return result


def _evidence_tier(rank: int, bounds: tuple[int, ...]) -> int:
    for index, upper in enumerate(bounds):
        if rank <= upper:
            return index
    return len(bounds)


def _prior_evidence_metadata(
    prior: TargetPrior, spec: ActiveNullSpec
) -> tuple[tuple[int, ...], bytes]:
    """Validate the sparse prior and return position-aligned rank metadata."""

    edge_count = len(prior.target_indices)
    if edge_count == 0:
        raise ContractError(
            "Active-null planning requires at least one target-prior link",
            code="active_null_empty_prior",
            field="target_prior",
            remediation="Provide a non-empty training-fold target prior",
        )
    inferred_ranks = [0] * edge_count if prior.ranks is None else None
    evidence_tiers: list[int] = []
    for driver_index in range(len(prior.driver_ids)):
        start = prior.indptr[driver_index]
        stop = prior.indptr[driver_index + 1]
        positions = range(start, stop)
        if prior.ranks is None:
            if inferred_ranks is None:  # pragma: no cover - type narrowing
                raise RuntimeError("rank inference storage is unavailable")
            ranked = sorted(
                positions,
                key=lambda position: (
                    -prior.weights[position],
                    prior.target_ids[prior.target_indices[position]],
                ),
            )
            for rank, position in enumerate(ranked, start=1):
                inferred_ranks[position] = rank
        seen_targets: set[int] = set()
        for position in positions:
            target_index = prior.target_indices[position]
            if target_index in seen_targets:
                raise ContractError(
                    "Target prior contains duplicate driver-target links",
                    code="active_null_duplicate_prior_link",
                    field="target_indices",
                    remediation="Deduplicate the source target prior before planning",
                )
            seen_targets.add(target_index)
            if inferred_ranks is not None:
                evidence_rank = inferred_ranks[position]
            else:
                source_ranks = prior.ranks
                if source_ranks is None:  # pragma: no cover - type narrowing
                    raise RuntimeError("source rank storage is unavailable")
                evidence_rank = source_ranks[position]
            if (
                isinstance(evidence_rank, bool)
                or not isinstance(evidence_rank, int)
                or evidence_rank < 1
            ):
                raise ContractError(
                    "Target prior ranks must be positive integers",
                    code="active_null_invalid_prior_rank",
                    field="ranks",
                    remediation=(
                        "Regenerate the target prior with valid positive ranks"
                    ),
                )
            for field_name, value in (
                ("driver_id", prior.driver_ids[driver_index]),
                ("target_id", prior.target_ids[target_index]),
            ):
                if not value or value != value.strip():
                    raise ValueError(
                        f"{field_name} must be a canonical non-empty string"
                    )
            evidence_tiers.append(
                _evidence_tier(evidence_rank, spec.rank_tier_upper_bounds)
            )
    evidence_ranks = (
        tuple(inferred_ranks) if inferred_ranks is not None else prior.ranks
    )
    if evidence_ranks is None:  # pragma: no cover - exhaustive narrowing
        raise RuntimeError("prior evidence ranks are unavailable")
    return evidence_ranks, bytes(evidence_tiers)


def _prior_links(
    prior: TargetPrior, spec: ActiveNullSpec
) -> tuple[ActiveNullPriorLink, ...]:
    evidence_ranks, evidence_tiers = _prior_evidence_metadata(prior, spec)
    links: list[ActiveNullPriorLink] = []
    for driver_index, driver_id in enumerate(prior.driver_ids):
        start = prior.indptr[driver_index]
        stop = prior.indptr[driver_index + 1]
        for position in range(start, stop):
            target_index = prior.target_indices[position]
            target_id = prior.target_ids[target_index]
            rank = None if prior.ranks is None else prior.ranks[position]
            links.append(
                ActiveNullPriorLink(
                    driver_id=driver_id,
                    target_id=target_id,
                    weight=prior.weights[position],
                    rank=rank,
                    evidence_rank=evidence_ranks[position],
                    evidence_tier=evidence_tiers[position],
                )
            )
    return tuple(links)


def _not_estimable_plan(
    *,
    spec: ActiveNullSpec,
    source_prior_content_id: str,
    seed_lineage: SeedLineage,
    source_edge_count: int,
    reason_code: ActiveNullReasonCode,
    diagnostics: tuple[ActiveNullTierDiagnostic, ...],
    source_overlap_count: int | None = None,
) -> ActiveNullPlan:
    overlap_fraction = (
        None
        if source_overlap_count is None
        else source_overlap_count / source_edge_count
    )
    return ActiveNullPlan(
        active_null_spec_id=spec.active_null_spec_id,
        source_prior_content_id=source_prior_content_id,
        seed_lineage=seed_lineage,
        status=ActiveNullPlanStatus.NOT_ESTIMABLE,
        reason_code=reason_code,
        source_edge_count=source_edge_count,
        accepted_switch_count=sum(item.accepted_switch_count for item in diagnostics),
        attempted_switch_count=sum(item.attempted_switch_count for item in diagnostics),
        source_overlap_count=source_overlap_count,
        source_overlap_fraction=overlap_fraction,
        tier_diagnostics=diagnostics,
        reassigned_links=(),
        _producer_token=_ACTIVE_NULL_PLAN_PRODUCER_TOKEN,
    )


def plan_active_null_target_prior(
    prior: TargetPrior,
    *,
    seed_lineage: SeedLineage,
    spec: ActiveNullSpec | None = None,
) -> ActiveNullPlan:
    """Plan one deterministic ADR-013 target-prior reassignment.

    The input prior must already represent the context-label-blind training-fold
    resource scope.  Infeasible switching or excessive residual overlap returns
    a typed not-estimable plan rather than an unchanged pseudo-null.
    """

    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    if not isinstance(seed_lineage, SeedLineage):
        raise TypeError("seed_lineage must be a SeedLineage")
    resolved_spec = ActiveNullSpec() if spec is None else spec
    if not isinstance(resolved_spec, ActiveNullSpec):
        raise TypeError("spec must be an ActiveNullSpec or None")
    expected_spec_id = stable_id(
        "active_null_spec",
        resolved_spec._identity_payload(),
        schema_version=_SPEC_SCHEMA_VERSION,
    )
    if resolved_spec.active_null_spec_id != expected_spec_id:
        raise ContractError(
            "Active-null specification identity is not intact",
            code="active_null_spec_integrity_violation",
            field="active_null_spec_id",
            remediation="Rebuild the accepted ADR-013 specification",
        )
    source_id = target_prior_content_id(prior)
    evidence_ranks, evidence_tiers = _prior_evidence_metadata(prior, resolved_spec)
    source_edge_count = len(evidence_tiers)
    target_count = len(prior.target_ids)
    mutable_target_indices = list(prior.target_indices)
    edge_driver_indices: list[int] = []
    source_targets_by_driver: list[set[int]] = []
    occupied: set[int] = set()
    for driver_index in range(len(prior.driver_ids)):
        start = prior.indptr[driver_index]
        stop = prior.indptr[driver_index + 1]
        driver_targets = set(prior.target_indices[start:stop])
        source_targets_by_driver.append(driver_targets)
        edge_driver_indices.extend([driver_index] * (stop - start))
        occupied.update(
            driver_index * target_count + target_index
            for target_index in prior.target_indices[start:stop]
        )
    if len(occupied) != source_edge_count:  # pragma: no cover - guarded upstream
        raise RuntimeError("active-null integer edge encoding is not unique")
    indices_by_tier = [
        array("Q") for _ in range(len(resolved_spec.rank_tier_upper_bounds) + 1)
    ]
    for index, tier in enumerate(evidence_tiers):
        indices_by_tier[tier].append(index)

    diagnostics: list[ActiveNullTierDiagnostic] = []
    for tier, indices in enumerate(indices_by_tier):
        if not indices:
            continue
        edge_count = len(indices)
        required = resolved_spec.accepted_switches_per_edge * edge_count
        maximum_attempts = required * resolved_spec.max_attempts_per_required_switch
        rng = seed_lineage.derive(
            "active-null-target-prior", f"evidence-tier-{tier}"
        ).python_random()
        accepted = 0
        attempts = 0
        while accepted < required and attempts < maximum_attempts:
            attempts += 1
            if edge_count < 2:
                break
            first_index, second_index = rng.sample(indices, 2)
            first_driver = edge_driver_indices[first_index]
            second_driver = edge_driver_indices[second_index]
            first_target = mutable_target_indices[first_index]
            second_target = mutable_target_indices[second_index]
            if first_driver == second_driver or first_target == second_target:
                continue
            first_old = first_driver * target_count + first_target
            second_old = second_driver * target_count + second_target
            first_new = first_driver * target_count + second_target
            second_new = second_driver * target_count + first_target
            # Distinct drivers and targets guarantee neither cross-edge can be
            # one of the two removed edges. Direct lookup is therefore exactly
            # equivalent to membership in ``occupied - {old_1, old_2}``.
            if first_new in occupied or second_new in occupied:
                continue
            occupied.remove(first_old)
            occupied.remove(second_old)
            occupied.add(first_new)
            occupied.add(second_new)
            mutable_target_indices[first_index] = second_target
            mutable_target_indices[second_index] = first_target
            accepted += 1
        tier_overlap = sum(
            mutable_target_indices[index]
            in source_targets_by_driver[edge_driver_indices[index]]
            for index in indices
        )
        diagnostics.append(
            ActiveNullTierDiagnostic(
                evidence_tier=tier,
                edge_count=edge_count,
                required_switch_count=required,
                accepted_switch_count=accepted,
                attempted_switch_count=attempts,
                source_overlap_count=tier_overlap,
            )
        )
        if accepted < required:
            return _not_estimable_plan(
                spec=resolved_spec,
                source_prior_content_id=source_id,
                seed_lineage=seed_lineage,
                source_edge_count=source_edge_count,
                reason_code=ActiveNullReasonCode.REASSIGNMENT_INFEASIBLE,
                diagnostics=tuple(diagnostics),
            )

    overlap_count = sum(
        target_index in source_targets_by_driver[driver_index]
        for driver_index, target_index in zip(
            edge_driver_indices, mutable_target_indices, strict=True
        )
    )
    overlap_fraction = overlap_count / source_edge_count
    if overlap_fraction > resolved_spec.max_overlap_fraction:
        return _not_estimable_plan(
            spec=resolved_spec,
            source_prior_content_id=source_id,
            seed_lineage=seed_lineage,
            source_edge_count=source_edge_count,
            reason_code=ActiveNullReasonCode.INSUFFICIENT_REWIRING,
            diagnostics=tuple(diagnostics),
            source_overlap_count=overlap_count,
        )
    reassigned_links = tuple(
        ActiveNullPriorLink(
            driver_id=prior.driver_ids[edge_driver_indices[index]],
            target_id=prior.target_ids[mutable_target_indices[index]],
            weight=prior.weights[index],
            rank=None if prior.ranks is None else prior.ranks[index],
            evidence_rank=evidence_ranks[index],
            evidence_tier=evidence_tiers[index],
        )
        for index in range(source_edge_count)
    )
    del (
        edge_driver_indices,
        evidence_ranks,
        evidence_tiers,
        indices_by_tier,
        mutable_target_indices,
        occupied,
        source_targets_by_driver,
    )
    return ActiveNullPlan(
        active_null_spec_id=resolved_spec.active_null_spec_id,
        source_prior_content_id=source_id,
        seed_lineage=seed_lineage,
        status=ActiveNullPlanStatus.OBSERVED,
        reason_code=None,
        source_edge_count=source_edge_count,
        accepted_switch_count=sum(item.accepted_switch_count for item in diagnostics),
        attempted_switch_count=sum(item.attempted_switch_count for item in diagnostics),
        source_overlap_count=overlap_count,
        source_overlap_fraction=overlap_fraction,
        tier_diagnostics=tuple(diagnostics),
        reassigned_links=reassigned_links,
        _producer_token=_ACTIVE_NULL_PLAN_PRODUCER_TOKEN,
    )


def _stub_signature(
    links: tuple[ActiveNullPriorLink, ...],
) -> Counter[tuple[str, float, int | None, int, int]]:
    return Counter(
        (
            link.driver_id,
            link.weight,
            link.rank,
            link.evidence_rank,
            link.evidence_tier,
        )
        for link in links
    )


def _target_tier_degree(
    links: tuple[ActiveNullPriorLink, ...],
) -> Counter[tuple[str, int]]:
    return Counter((link.target_id, link.evidence_tier) for link in links)


def materialize_active_null_target_prior(
    prior: TargetPrior, plan: ActiveNullPlan
) -> TargetPrior:
    """Materialize a verified observed plan without mutating its source prior."""

    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    if not isinstance(plan, ActiveNullPlan):
        raise TypeError("plan must be an ActiveNullPlan")
    plan.require_intact()
    source_id = target_prior_content_id(prior)
    if source_id != plan.source_prior_content_id:
        raise ContractError(
            "Active-null plan does not belong to this target prior",
            code="active_null_source_mismatch",
            field="source_prior_content_id",
            remediation="Replan from the exact training-fold target prior",
        )
    if plan.status is not ActiveNullPlanStatus.OBSERVED:
        raise ContractError(
            "A not-estimable active-null plan cannot be materialized",
            code="active_null_plan_not_estimable",
            field="status",
            remediation="Generate a feasible observed reassignment plan",
        )
    spec = ActiveNullSpec()
    if plan.active_null_spec_id != spec.active_null_spec_id:
        raise ContractError(
            "Active-null plan uses an unsupported specification",
            code="active_null_spec_mismatch",
            field="active_null_spec_id",
            remediation="Regenerate the plan with the accepted ADR-013 spec",
        )
    source_links = _prior_links(prior, spec)
    reassigned = plan.reassigned_links
    if (
        len(reassigned) != len(source_links)
        or Counter(link.driver_id for link in reassigned)
        != Counter(link.driver_id for link in source_links)
        or _target_tier_degree(reassigned) != _target_tier_degree(source_links)
        or _stub_signature(reassigned) != _stub_signature(source_links)
    ):
        raise ContractError(
            "Active-null plan violates degree or evidence preservation",
            code="active_null_invariant_violation",
            field="reassigned_links",
            remediation="Reject the plan and regenerate it from the source prior",
        )
    if any(
        link.driver_id not in set(prior.driver_ids)
        or link.target_id not in set(prior.target_ids)
        for link in reassigned
    ):
        raise ContractError(
            "Active-null plan contains an entity outside the source prior",
            code="active_null_universe_mismatch",
            field="reassigned_links",
            remediation="Regenerate the plan from the exact source universe",
        )
    source_edges = {(link.driver_id, link.target_id) for link in source_links}
    reassigned_edges = {(link.driver_id, link.target_id) for link in reassigned}
    if len(reassigned_edges) != len(reassigned):
        raise ContractError(
            "Active-null plan contains duplicate driver-target links",
            code="active_null_invariant_violation",
            field="reassigned_links",
            remediation="Reject the plan and regenerate it",
        )
    overlap_count = len(source_edges.intersection(reassigned_edges))
    if (
        overlap_count != plan.source_overlap_count
        or overlap_count / len(source_links) > spec.max_overlap_fraction
    ):
        raise ContractError(
            "Active-null plan overlap diagnostics are not intact",
            code="active_null_invariant_violation",
            field="source_overlap_count",
            remediation="Reject the plan and regenerate it",
        )

    target_index = {target: index for index, target in enumerate(prior.target_ids)}
    by_driver: dict[str, list[ActiveNullPriorLink]] = defaultdict(list)
    for link in reassigned:
        by_driver[link.driver_id].append(link)
    target_indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    indptr = [0]
    for driver_id in prior.driver_ids:
        driver_links = sorted(
            by_driver[driver_id],
            key=lambda link: (
                link.evidence_rank,
                link.target_id,
                link.weight,
            ),
        )
        target_indices.extend(target_index[link.target_id] for link in driver_links)
        weights.extend(link.weight for link in driver_links)
        ranks.extend(link.rank for link in driver_links if link.rank is not None)
        indptr.append(len(target_indices))
    notes = tuple(
        sorted(
            {
                *prior.mapping_report.notes,
                f"active_null_plan_id={plan.active_null_plan_id}",
                f"active_null_source_prior_content_id={source_id}",
            }
        )
    )
    mapping_report: MappingReport = replace(prior.mapping_report, notes=notes)
    manifest_digest = stable_id(
        "active_null_target_prior_manifest",
        {
            "source_manifest_digest": prior.manifest_digest,
            "source_prior_content_id": source_id,
            "active_null_plan_id": plan.active_null_plan_id,
            "active_null_spec_id": plan.active_null_spec_id,
        },
    )
    return TargetPrior(
        resource_id=prior.resource_id,
        version=prior.version,
        species=prior.species,
        gene_namespace=prior.gene_namespace,
        driver_kind=prior.driver_kind,
        target_ids=prior.target_ids,
        driver_ids=prior.driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(weights),
        ranks=None if prior.ranks is None else tuple(ranks),
        direction=prior.direction,
        evidence=prior.evidence,
        mapping_report=mapping_report,
        manifest_digest=manifest_digest,
    )


__all__ = [
    "ActiveNullPlan",
    "ActiveNullPlanStatus",
    "ActiveNullPriorLink",
    "ActiveNullReasonCode",
    "ActiveNullSpec",
    "ActiveNullTierDiagnostic",
    "materialize_active_null_target_prior",
    "plan_active_null_target_prior",
    "target_prior_content_id",
]
