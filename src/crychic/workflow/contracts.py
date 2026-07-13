"""Typed orchestration artifacts for the v0.1 exploratory baseline."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from crychic.attribution import AttributionResult, GatedTargetBasis
from crychic.availability import BatchAvailability
from crychic.core import CrychicConfig, canonical_digest, canonical_json
from crychic.data import InputSchema, ValidatedInput
from crychic.design import ContextGraph, DesignAudit
from crychic.pseudobulk import ExploratoryAggregate, PseudobulkDataset
from crychic.resources import ResourceBundle, TargetPrior
from crychic.response import ResponseEstimate
from crychic.scoring import (
    CommunicationScores,
    ScoringCollectionManifest,
    ScoringFunctional,
)
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


EDGE_EVIDENCE_PRIMARY_KEY = (
    "contrast",
    "fold_id",
    "sample_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "mode",
)
EDGE_EVIDENCE_LINK_KEY = (
    "contrast",
    "fold_id",
    "sample_id",
    "subject_id",
    "context",
    "sender",
    "receiver",
    "interaction_id",
    "mode",
)
EDGE_EVIDENCE_COLUMNS = (
    "sample_id",
    "subject_id",
    "context",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "driver_id",
    "mode",
    "contrast",
    "fold_id",
    "state_availability",
    "state_availability_status",
    "state_availability_reason_code",
    "ecosystem_availability",
    "ecosystem_availability_status",
    "ecosystem_availability_reason_code",
    "receptor_gate",
    "receptor_gate_status",
    "receptor_gate_reason_code",
    "receiver_program_score",
    "receiver_program_status",
    "receiver_program_reason_code",
    "incremental_downstream",
    "incremental_downstream_status",
    "incremental_downstream_reason_code",
    "attribution_support",
    "attribution_support_method",
    "attribution_support_status",
    "attribution_support_reason_code",
    "sender_weight",
    "sender_status",
    "sender_reason_code",
    "prior_quality",
    "prior_quality_source",
    "prior_quality_status",
    "prior_quality_reason_code",
    "legacy_downstream_activity",
    "legacy_downstream_status",
    "legacy_downstream_reason_code",
    "legacy_integrated_strength",
    "legacy_integrated_status",
    "legacy_integrated_reason_code",
    "scoring_function_status",
    "scoring_function_reason_code",
    "score_version",
    "model_manifest_id",
    "scoring_function_id",
    "sample_score_status",
    "sample_score_reason_code",
)

_EDGE_EVIDENCE_VALUE_CONTRACTS = (
    (
        "state_availability",
        "state_availability_status",
        "state_availability_reason_code",
        frozenset({"observed"}),
    ),
    (
        "ecosystem_availability",
        "ecosystem_availability_status",
        "ecosystem_availability_reason_code",
        frozenset({"observed"}),
    ),
    (
        "receptor_gate",
        "receptor_gate_status",
        "receptor_gate_reason_code",
        frozenset({"observed"}),
    ),
    (
        "receiver_program_score",
        "receiver_program_status",
        "receiver_program_reason_code",
        frozenset({"observed"}),
    ),
    (
        "incremental_downstream",
        "incremental_downstream_status",
        "incremental_downstream_reason_code",
        frozenset({"observed"}),
    ),
    (
        "attribution_support",
        "attribution_support_status",
        "attribution_support_reason_code",
        frozenset({"observed"}),
    ),
    (
        "sender_weight",
        "sender_status",
        "sender_reason_code",
        frozenset({"ok"}),
    ),
    (
        "prior_quality",
        "prior_quality_status",
        "prior_quality_reason_code",
        frozenset({"observed"}),
    ),
    (
        "legacy_downstream_activity",
        "legacy_downstream_status",
        "legacy_downstream_reason_code",
        frozenset({"observed", "zero_attribution"}),
    ),
    (
        "legacy_integrated_strength",
        "legacy_integrated_status",
        "legacy_integrated_reason_code",
        frozenset({"ok"}),
    ),
)


@dataclass(frozen=True, slots=True)
class EdgeEvidenceLedger:
    """Component-level sample edge evidence with explicit missingness semantics."""

    table: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("edge evidence must be a pandas DataFrame")
        table = self.table.copy(deep=True)
        missing = set(EDGE_EVIDENCE_COLUMNS).difference(table.columns)
        extra = set(table.columns).difference(EDGE_EVIDENCE_COLUMNS)
        if missing or extra:
            raise ValueError(
                "edge evidence columns do not match the workflow contract: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if table.duplicated(list(EDGE_EVIDENCE_PRIMARY_KEY)).any():
            raise ValueError("edge evidence primary keys must be unique")
        identifier_columns = (
            "sample_id",
            "subject_id",
            "context",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "contrast",
            "fold_id",
        )
        for column in identifier_columns:
            if table[column].isna().any():
                raise ValueError(f"edge evidence {column} must not contain NA")
            if any(isinstance(value, str) and not value for value in table[column]):
                raise ValueError(
                    f"edge evidence {column} must not contain empty identifiers"
                )
        if not set(table["mode"]).issubset({"state", "ecosystem"}):
            raise ValueError("edge evidence mode must be state or ecosystem")
        if table.groupby("sample_id", dropna=False)["subject_id"].nunique().gt(1).any():
            raise ValueError("each edge evidence sample_id must map to one subject_id")

        for (
            value_column,
            status_column,
            reason_column,
            success_statuses,
        ) in _EDGE_EVIDENCE_VALUE_CONTRACTS:
            numeric = pd.to_numeric(table[value_column], errors="coerce")
            invalid_numeric = table[value_column].notna() & numeric.isna()
            present = numeric.dropna().to_numpy(dtype=float)
            if invalid_numeric.any() or (
                present.size
                and (
                    not np.isfinite(present).all()
                    or ((present < 0) | (present > 1)).any()
                )
            ):
                raise ValueError(
                    f"edge evidence {value_column} must contain "
                    "unit-interval values or NA"
                )
            if table[status_column].isna().any():
                raise ValueError(f"edge evidence {status_column} must not contain NA")
            value_missing = numeric.isna()
            reason_missing = table[reason_column].isna()
            if (value_missing & reason_missing).any():
                raise ValueError(
                    f"edge evidence null {value_column} requires {reason_column}"
                )
            success = table[status_column].isin(success_statuses)
            if (success & (value_missing | ~reason_missing)).any():
                raise ValueError(
                    f"edge evidence successful {status_column} requires a value "
                    "and no reason"
                )
            if ((~success) & reason_missing).any():
                raise ValueError(
                    f"edge evidence degraded {status_column} requires {reason_column}"
                )
            unavailable = (
                table[status_column]
                .astype(str)
                .isin(
                    {
                        "missing",
                        "missing_evidence",
                        "missing_core_evidence",
                        "not_estimable",
                        "unavailable",
                        "failed",
                        "abundance_not_estimable",
                    }
                )
            )
            if (unavailable & ~value_missing).any():
                raise ValueError(
                    f"edge evidence unavailable {status_column} requires an NA value"
                )

        methods = table["attribution_support_method"]
        if methods.isna().any() or any(
            not isinstance(value, str) or not value for value in methods
        ):
            raise ValueError(
                "edge evidence attribution_support_method must contain identifiers"
            )
        prior_present = table["prior_quality"].notna()
        if (prior_present & table["prior_quality_source"].isna()).any():
            raise ValueError("observed prior quality requires prior_quality_source")

        provenance_columns = (
            "score_version",
            "model_manifest_id",
            "scoring_function_id",
        )
        provenance_count = pd.Series(0, index=table.index, dtype="int64")
        for column in provenance_columns:
            provenance_count = provenance_count + table[column].notna().astype("int64")
        any_provenance = provenance_count.gt(0)
        all_provenance = provenance_count.eq(len(provenance_columns))
        if (any_provenance != all_provenance).any():
            raise ValueError("score provenance must be entirely present or entirely NA")
        no_provenance = ~all_provenance
        if (no_provenance & table["scoring_function_reason_code"].isna()).any():
            raise ValueError("missing score provenance requires a scoring reason")
        if table["scoring_function_status"].isna().any():
            raise ValueError("scoring_function_status must not contain NA")

        allowed_links = {"linked", "not_emitted"}
        if not set(table["sample_score_status"]).issubset(allowed_links):
            raise ValueError("sample_score_status must be linked or not_emitted")
        linked = table["sample_score_status"].eq("linked")
        link_reason_missing = table["sample_score_reason_code"].isna()
        if (linked & ~link_reason_missing).any() or (
            (~linked) & link_reason_missing
        ).any():
            raise ValueError(
                "sample score link status and sample_score_reason_code disagree"
            )
        object.__setattr__(self, "table", table)


def _validate_edge_evidence_links(
    edge_evidence: pd.DataFrame, sample_scores: pd.DataFrame
) -> None:
    linked = edge_evidence.loc[edge_evidence["sample_score_status"].eq("linked")].copy()
    if sample_scores.empty and linked.empty:
        return
    if sample_scores.duplicated(list(EDGE_EVIDENCE_LINK_KEY)).any():
        raise ValueError("baseline sample score link keys must be unique")
    ledger_view = linked.loc[:, list(EDGE_EVIDENCE_LINK_KEY)].copy()
    ledger_view["availability"] = np.where(
        linked["mode"].eq("state"),
        linked["state_availability"],
        linked["ecosystem_availability"],
    )
    ledger_view["downstream_activity"] = linked["legacy_downstream_activity"]
    ledger_view["sender_component"] = pd.to_numeric(
        linked["sender_weight"], errors="coerce"
    ).where(linked["sender_status"].eq("ok"))
    ledger_view["prior_quality"] = linked["prior_quality"]
    ledger_view["comm_strength"] = linked["legacy_integrated_strength"]
    ledger_view["status"] = linked["legacy_integrated_status"]
    ledger_view["reason_code"] = linked["legacy_integrated_reason_code"]
    ledger_view["functional_status"] = linked["scoring_function_status"]
    ledger_view["functional_reason_code"] = linked["scoring_function_reason_code"]
    for column in ("score_version", "model_manifest_id", "scoring_function_id"):
        ledger_view[column] = linked[column]

    comparison_columns = (
        "availability",
        "downstream_activity",
        "sender_component",
        "prior_quality",
        "comm_strength",
        "status",
        "reason_code",
        "functional_status",
        "functional_reason_code",
        "score_version",
        "model_manifest_id",
        "scoring_function_id",
    )
    score_view = sample_scores.loc[
        :, [*EDGE_EVIDENCE_LINK_KEY, *comparison_columns]
    ].copy()
    merged = score_view.merge(
        ledger_view,
        how="outer",
        on=list(EDGE_EVIDENCE_LINK_KEY),
        suffixes=("_score", "_ledger"),
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "every baseline sample score must link to exactly one edge evidence row"
        )
    for column in (
        "availability",
        "downstream_activity",
        "sender_component",
        "prior_quality",
        "comm_strength",
    ):
        score_values = pd.to_numeric(
            merged[f"{column}_score"], errors="coerce"
        ).to_numpy(dtype=float)
        ledger_values = pd.to_numeric(
            merged[f"{column}_ledger"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.allclose(
            score_values,
            ledger_values,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"edge evidence does not reproduce sample score {column}")
    for column in (
        "status",
        "reason_code",
        "functional_status",
        "functional_reason_code",
        "score_version",
        "model_manifest_id",
        "scoring_function_id",
    ):
        score_text = merged[f"{column}_score"].astype("string").fillna("<NA>")
        ledger_text = merged[f"{column}_ledger"].astype("string").fillna("<NA>")
        if not score_text.equals(ledger_text):
            raise ValueError(f"edge evidence does not reproduce sample score {column}")


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
    precision_method: str = "not_applied"
    precision_transform_id: str | None = None
    precision_lower_quantile: float | None = None
    precision_upper_quantile: float | None = None
    n_positive_precision_features: int = 0
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
        if status is RunStatus.OK and (self.basis is None or self.attribution is None):
            raise ValueError("status=ok requires basis and attribution artifacts")
        if status is not RunStatus.OK and not self.reason_code:
            raise ValueError("non-success attribution requires an explicit reason_code")
        if not self.precision_method.strip():
            raise ValueError("precision_method must be a non-empty string")
        if self.precision_transform_id is None:
            if (
                self.precision_lower_quantile is not None
                or self.precision_upper_quantile is not None
                or self.n_positive_precision_features != 0
            ):
                raise ValueError(
                    "precision diagnostics require a precision_transform_id"
                )
        else:
            if self.precision_method == "not_applied":
                raise ValueError("applied precision transform requires its method name")
            lower = self.precision_lower_quantile
            upper = self.precision_upper_quantile
            if (
                lower is None
                or upper is None
                or not 0 <= lower <= upper <= 1
                or self.n_positive_precision_features < 0
            ):
                raise ValueError("precision transform diagnostics are invalid")
        if status is RunStatus.OK and self.precision_transform_id is None:
            raise ValueError("successful attribution requires precision provenance")
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
    edge_evidence: pd.DataFrame
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
        edge_evidence = EdgeEvidenceLedger(self.edge_evidence).table
        _validate_edge_evidence_links(edge_evidence, self.sample_scores)
        canonical_json(self.run_parameters)
        object.__setattr__(self, "sample_scores", self.sample_scores.copy(deep=True))
        object.__setattr__(self, "edge_evidence", edge_evidence)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        object.__setattr__(self, "run_parameters", _freeze(self.run_parameters))

    @property
    def run_parameters_digest(self) -> str:
        """Digest effective workflow choices and fitted filter provenance."""

        return canonical_digest(self.run_parameters)

    @property
    def inference_eligible(self) -> bool:
        """The v0.1 workflow is descriptive regardless of input mode."""

        return False

    @property
    def scoring_functionals(self) -> tuple[ScoringFunctional, ...]:
        """Return each receiver branch's context-common functional."""

        return tuple(run.functional for run in self.score_runs)

    @property
    def scoring_collections(self) -> tuple[ScoringCollectionManifest, ...]:
        """Return complete receiver partitions for persisted score branches."""

        if not self.score_runs:
            return ()
        from .persistence import baseline_scoring_collections

        return baseline_scoring_collections(self)

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
