"""Typed orchestration artifacts for the v0.1 exploratory baseline."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd

from crychic.attribution import AttributionResult, GatedTargetBasis
from crychic.availability import BatchAvailability
from crychic.core import CrychicConfig, canonical_digest, canonical_json
from crychic.data import InputSchema, ValidatedInput
from crychic.design import ContextGraph, DesignAudit
from crychic.pseudobulk import ExploratoryAggregate, PseudobulkDataset
from crychic.resources import ResourceBundle, TargetPrior
from crychic.response import ResponseEstimate
from crychic.scoring import CommunicationScores, ScoringFunctional
from crychic.sender import SenderAssignment


class PlanStatus(StrEnum):
    """Readiness of one dry-run stage."""

    READY = "ready"
    OPTIONAL = "optional"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class RunStatus(StrEnum):
    """Terminal state of one receiver/contrast workflow branch."""

    OK = "ok"
    NOT_ESTIMABLE = "not_estimable"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class BaselineMode(StrEnum):
    """Highest-level quantity produced by a baseline run."""

    EXPLORATORY_STRENGTH = "exploratory_strength"
    AVAILABILITY_BASELINE = "availability_baseline"


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One dry-run stage and its current readiness."""

    name: str
    status: PlanStatus
    detail: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.detail.strip():
            raise ValueError("stage plan name and detail must be non-empty")
        object.__setattr__(self, "status", PlanStatus(self.status))


@dataclass(frozen=True, slots=True)
class BaselineDryRunPlan:
    """Resource, context, and sample-support plan before matrix aggregation."""

    config_digest: str
    input_schema: InputSchema
    context_graph: ContextGraph
    design_audit: DesignAudit
    contrast_table: pd.DataFrame
    support_table: pd.DataFrame
    resource_table: pd.DataFrame
    stages: tuple[StagePlan, ...]
    warnings: tuple[str, ...]
    can_fit: bool

    def __post_init__(self) -> None:
        if not self.config_digest:
            raise ValueError("config_digest must be non-empty")
        if not self.stages:
            raise ValueError("dry-run plan must contain at least one stage")
        object.__setattr__(self, "support_table", self.support_table.copy(deep=True))
        object.__setattr__(self, "resource_table", self.resource_table.copy(deep=True))
        object.__setattr__(self, "contrast_table", self.contrast_table.copy(deep=True))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))


@dataclass(frozen=True, slots=True)
class BaselineAttributionRun:
    """One response-aligned experimental attribution attempt."""

    receiver: Hashable
    contrast: str
    contrast_contexts: tuple[Hashable, ...]
    status: RunStatus
    reason_code: str | None
    receptor_gates: tuple[tuple[str, float], ...]
    driver_by_interaction: tuple[tuple[str, str], ...]
    basis: GatedTargetBasis | None
    attribution: AttributionResult | None
    response_field: str = "effect"
    precision_method: str = "uniform"
    experimental: bool = True

    def __post_init__(self) -> None:
        status = RunStatus(self.status)
        if not self.contrast.strip() or len(set(self.contrast_contexts)) < 2:
            raise ValueError("attribution run requires a named multi-context contrast")
        if len({driver for driver, _ in self.receptor_gates}) != len(
            self.receptor_gates
        ):
            raise ValueError("receptor gate driver IDs must be unique")
        if any(not 0 <= gate <= 1 for _, gate in self.receptor_gates):
            raise ValueError("receptor gates must lie in [0, 1]")
        if status is RunStatus.OK and (
            self.basis is None or self.attribution is None
        ):
            raise ValueError("status=ok requires basis and attribution artifacts")
        if status is not RunStatus.OK and not self.reason_code:
            raise ValueError("non-success attribution requires an explicit reason_code")
        if not self.experimental:
            raise ValueError("v0.1 attribution must remain experimental")
        object.__setattr__(self, "status", status)

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.OK


@dataclass(frozen=True, slots=True)
class BaselineScoreRun:
    """One receiver/contrast common-functional scoring application."""

    receiver: Hashable
    contrast: str
    functional: ScoringFunctional
    downstream_activity: pd.DataFrame
    scores: CommunicationScores

    def __post_init__(self) -> None:
        if self.contrast != self.functional.contrast_name:
            raise ValueError("score-run contrast must match its scoring functional")
        if self.scores.functional.scoring_function_id != (
            self.functional.scoring_function_id
        ):
            raise ValueError("score run must retain its exact scoring functional")
        object.__setattr__(
            self, "downstream_activity", self.downstream_activity.copy(deep=True)
        )


@dataclass(frozen=True, slots=True)
class BaselineArtifacts:
    """Complete in-memory result of one v0.1 baseline orchestration."""

    config: CrychicConfig
    input_schema: InputSchema
    validated_input: ValidatedInput
    aggregate: PseudobulkDataset | ExploratoryAggregate
    context_graph: ContextGraph
    response: ResponseEstimate
    resource_bundle: ResourceBundle
    target_prior: TargetPrior | None
    availability: BatchAvailability
    sender_assignment: SenderAssignment
    dry_run_plan: BaselineDryRunPlan
    attribution_runs: tuple[BaselineAttributionRun, ...]
    score_runs: tuple[BaselineScoreRun, ...]
    sample_scores: pd.DataFrame
    mode: BaselineMode
    reason_codes: tuple[str, ...]
    run_parameters: Mapping[str, object]
    method_version: str = "0.1.0-exploratory"

    def __post_init__(self) -> None:
        mode = BaselineMode(self.mode)
        if mode is BaselineMode.EXPLORATORY_STRENGTH and not self.score_runs:
            raise ValueError("exploratory_strength mode requires score runs")
        if mode is BaselineMode.AVAILABILITY_BASELINE and self.score_runs:
            raise ValueError("availability_baseline cannot contain integrated scores")
        forbidden_names = {
            "p",
            "p_value",
            "q",
            "q_value",
            "posterior",
            "posterior_probability",
        }
        forbidden = {
            str(column).lower()
            for column in self.sample_scores.columns
            if str(column).lower() in forbidden_names
            or "probability" in str(column).lower()
        }
        if forbidden:
            raise ValueError(
                "baseline sample scores cannot contain inferential fields: "
                f"{sorted(forbidden)}"
            )
        canonical_json(self.run_parameters)
        object.__setattr__(self, "sample_scores", self.sample_scores.copy(deep=True))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        object.__setattr__(self, "run_parameters", _freeze(self.run_parameters))

    @property
    def run_parameters_digest(self) -> str:
        """Digest every effective non-data workflow choice."""

        return canonical_digest(self.run_parameters)

    @property
    def inference_eligible(self) -> bool:
        """The v0.1 workflow is descriptive regardless of input mode."""

        return False

    @property
    def scoring_functionals(self) -> tuple[ScoringFunctional, ...]:
        """Return the frozen common functional for every scored branch."""

        return tuple(run.functional for run in self.score_runs)

    @property
    def downstream_activity(self) -> pd.DataFrame:
        """Return all receiver/contrast downstream rows with their provenance."""

        if not self.score_runs:
            return pd.DataFrame()
        return pd.concat(
            [run.downstream_activity for run in self.score_runs],
            ignore_index=True,
            sort=False,
        )

    @property
    def score_semantics(self) -> str:
        return (
            "strength_not_probability"
            if self.score_runs
            else "availability_not_probability"
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
