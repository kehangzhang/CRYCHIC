"""Producer-owned contracts for exploratory sender evidence assignment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import pandas as pd

from crychic.core import ContractError, stable_id

SENDER_EVIDENCE_COMPONENTS = (
    "ligand_availability",
    "cell_type_specificity",
    "subject_prevalence",
    "adjusted_coupling",
)
ASSIGNMENT_GROUP_COLUMNS = ("context_id", "receiver", "interaction_id")
SENDER_ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "assignment_functional_id",
    "context_id",
    "receiver",
    "interaction_id",
    "sender",
    "ligand_availability",
    "cell_type_specificity",
    "subject_prevalence",
    "adjusted_coupling",
    "coupling_weight",
    "evidence_score",
    "assignment_weight",
    "normalized_entropy",
    "n_subjects",
    "n_group_subjects",
    "status",
    "reason_code",
    "coupling_status",
    "coupling_reason_code",
    "assignment_mode",
)

_COUPLING_REASON = "v0_1_adjusted_coupling_disabled"
_ASSIGNMENT_MODE = "exploratory_in_sample"
_FORBIDDEN_INFERENCE_COLUMNS = {
    "p",
    "p_value",
    "q",
    "q_value",
    "posterior",
    "posterior_probability",
}


class SenderAssignmentStatus(StrEnum):
    """Estimability of one candidate sender's descriptive evidence."""

    OK = "ok"
    LOW_SUPPORT = "low_support"
    PARTIAL_EVIDENCE = "partial_evidence"
    MISSING_EVIDENCE = "missing_evidence"


class SenderCouplingStatus(StrEnum):
    """Availability of covariate-adjusted sender/receiver coupling."""

    NOT_ESTIMABLE_V0_1 = "not_estimable_v0_1"


def _default_component_weights() -> dict[str, float]:
    return {
        "ligand_availability": 1.0,
        "cell_type_specificity": 1.0,
        "subject_prevalence": 1.0,
        "adjusted_coupling": 0.0,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class SenderEvidenceParameters:
    """Frozen v0.1 parameters for descriptive sender evidence and softmax."""

    min_subjects: int = 3
    prevalence_threshold: float = 0.0
    softmax_temperature: float = 1.0
    component_weights: Mapping[str, float] = field(
        default_factory=_default_component_weights
    )
    schema_version: str = "0.1.0"
    frozen: bool = True
    assignment_functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_subjects, bool)
            or not isinstance(self.min_subjects, int)
            or self.min_subjects < 1
        ):
            raise ContractError(
                "min_subjects must be a positive integer",
                code="invalid_sender_parameters",
                field="min_subjects",
                remediation="Choose a positive subject-support threshold",
            )
        threshold = float(self.prevalence_threshold)
        temperature = float(self.softmax_temperature)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ContractError(
                "prevalence_threshold must lie in [0, 1]",
                code="invalid_sender_parameters",
                field="prevalence_threshold",
                remediation="Use a ligand-availability threshold on the unit interval",
            )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ContractError(
                "softmax_temperature must be finite and positive",
                code="invalid_sender_parameters",
                field="softmax_temperature",
                remediation="Use a fixed positive exploratory temperature",
            )
        if set(self.component_weights) != set(SENDER_EVIDENCE_COMPONENTS):
            raise ContractError(
                "component_weights must define every sender evidence component",
                code="invalid_sender_parameters",
                field="component_weights",
                remediation=f"Define exactly {list(SENDER_EVIDENCE_COMPONENTS)}",
            )
        weights: dict[str, float] = {}
        for component in SENDER_EVIDENCE_COMPONENTS:
            weight = float(self.component_weights[component])
            if not math.isfinite(weight) or weight < 0:
                raise ContractError(
                    "sender component weights must be finite and non-negative",
                    code="invalid_sender_parameters",
                    field="component_weights",
                    remediation="Use fixed non-negative evidence weights",
                )
            weights[component] = weight
        if weights["adjusted_coupling"] != 0:
            raise ContractError(
                "v0.1 adjusted coupling must have weight zero",
                code="coupling_not_available_v0_1",
                field="component_weights",
                remediation=(
                    "Set adjusted_coupling to 0 until a validated fold model exists"
                ),
            )
        if not any(
            weights[component] > 0
            for component in SENDER_EVIDENCE_COMPONENTS
            if component != "adjusted_coupling"
        ):
            raise ContractError(
                "at least one descriptive sender component must have positive weight",
                code="invalid_sender_parameters",
                field="component_weights",
                remediation="Enable ligand, specificity, or prevalence evidence",
            )
        if self.schema_version != "0.1.0" or not self.frozen:
            raise ContractError(
                "v0.1 sender parameters must use schema 0.1.0 and be frozen",
                code="invalid_sender_parameters",
                field="schema_version",
                remediation="Use the released immutable v0.1 parameter contract",
            )
        functional_id = stable_id(
            "sender_assignment_functional",
            {
                "component_weights": [
                    [component, weights[component]]
                    for component in SENDER_EVIDENCE_COMPONENTS
                ],
                "min_subjects": self.min_subjects,
                "prevalence_threshold": threshold,
                "schema_version": self.schema_version,
                "softmax_temperature": temperature,
            },
            schema_version=self.schema_version,
        )
        object.__setattr__(self, "prevalence_threshold", threshold)
        object.__setattr__(self, "softmax_temperature", temperature)
        object.__setattr__(self, "component_weights", MappingProxyType(weights))
        object.__setattr__(self, "assignment_functional_id", functional_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready parameter definition."""

        return {
            "assignment_functional_id": self.assignment_functional_id,
            "min_subjects": self.min_subjects,
            "prevalence_threshold": self.prevalence_threshold,
            "softmax_temperature": self.softmax_temperature,
            "component_weights": dict(self.component_weights),
            "schema_version": self.schema_version,
            "frozen": self.frozen,
            "assignment_mode": _ASSIGNMENT_MODE,
        }


def sender_assignment_id(
    *,
    functional_id: str,
    context_id: str,
    receiver: str,
    interaction_id: str,
    sender: str,
    schema_version: str = "0.1.0",
) -> str:
    """Build the stable ID for one sender candidate at its declared grain."""

    return stable_id(
        "sender_assignment",
        {
            "assignment_functional_id": functional_id,
            "context_id": context_id,
            "interaction_id": interaction_id,
            "receiver": receiver,
            "sender": sender,
        },
        schema_version=schema_version,
    )


def _numeric_interval(table: pd.DataFrame, column: str, *, nullable: bool) -> pd.Series:
    numeric = pd.to_numeric(table[column], errors="coerce")
    invalid_type = table[column].notna() & numeric.isna()
    if invalid_type.any() or (not nullable and numeric.isna().any()):
        raise ContractError(
            f"{column} must contain unit-interval numeric values"
            + (" or NA" if nullable else ""),
            code="invalid_sender_assignment",
            field=column,
            remediation="Build the table through assign_senders",
        )
    present = [float(value) for value in numeric.dropna().tolist()]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in present):
        raise ContractError(
            f"{column} values must lie in [0, 1]",
            code="invalid_sender_assignment",
            field=column,
            remediation="Preserve the released sender evidence scale",
        )
    return numeric


@dataclass(frozen=True, slots=True)
class SenderAssignment:
    """Validated, descriptive soft assignment of candidate sender cell types."""

    table: pd.DataFrame
    parameters: SenderEvidenceParameters

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, SenderEvidenceParameters):
            raise TypeError("parameters must be SenderEvidenceParameters")
        table = self.table.copy(deep=True)
        forbidden = {
            str(column)
            for column in table.columns
            if str(column).lower() in _FORBIDDEN_INFERENCE_COLUMNS
            or "probability" in str(column).lower()
        }
        if forbidden:
            raise ContractError(
                "sender assignment cannot contain inferential fields: "
                f"{sorted(forbidden)}",
                code="forbidden_sender_inference",
                field="columns",
                remediation=(
                    "Keep p/q/posterior quantities out of exploratory assignment"
                ),
            )
        missing = set(SENDER_ASSIGNMENT_COLUMNS).difference(table.columns)
        extra = set(table.columns).difference(SENDER_ASSIGNMENT_COLUMNS)
        if missing or extra:
            raise ContractError(
                "sender assignment columns do not match the producer contract",
                code="invalid_sender_assignment",
                field="columns",
                remediation=f"Use exactly {list(SENDER_ASSIGNMENT_COLUMNS)}",
            )
        identifier_columns = (
            "assignment_id",
            "assignment_functional_id",
            "context_id",
            "receiver",
            "interaction_id",
            "sender",
        )
        for column in identifier_columns:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in table[column].tolist()
            ):
                raise ContractError(
                    f"{column} must contain non-empty string identifiers",
                    code="invalid_sender_assignment",
                    field=column,
                    remediation="Use stable identifiers at the sender assignment grain",
                )
        key = [*ASSIGNMENT_GROUP_COLUMNS, "sender"]
        if table.duplicated(key).any() or table["assignment_id"].duplicated().any():
            raise ContractError(
                "sender assignment primary keys must be unique",
                code="duplicate_sender_assignment",
                field="assignment_id",
                remediation=(
                    "Emit one candidate row per context/receiver/interaction/sender"
                ),
            )
        if set(table["assignment_functional_id"]) not in (
            set(),
            {self.parameters.assignment_functional_id},
        ):
            raise ContractError(
                "sender rows do not match the supplied assignment functional",
                code="invalid_sender_assignment",
                field="assignment_functional_id",
                remediation="Keep parameters and their produced table together",
            )
        for row in table.itertuples(index=False):
            expected_id = sender_assignment_id(
                functional_id=self.parameters.assignment_functional_id,
                context_id=cast(str, row.context_id),
                receiver=cast(str, row.receiver),
                interaction_id=cast(str, row.interaction_id),
                sender=cast(str, row.sender),
                schema_version=self.parameters.schema_version,
            )
            if row.assignment_id != expected_id:
                raise ContractError(
                    "sender assignment ID does not match its stable grain",
                    code="invalid_sender_assignment",
                    field="assignment_id",
                    remediation="Generate assignment IDs with the producer constructor",
                )
        numeric_columns = {
            column: _numeric_interval(
                table,
                column,
                nullable=column
                in {
                    "ligand_availability",
                    "cell_type_specificity",
                    "subject_prevalence",
                    "evidence_score",
                    "assignment_weight",
                    "normalized_entropy",
                },
            )
            for column in (
                "ligand_availability",
                "cell_type_specificity",
                "subject_prevalence",
                "coupling_weight",
                "evidence_score",
                "assignment_weight",
                "normalized_entropy",
            )
        }
        if table["adjusted_coupling"].notna().any():
            raise ContractError(
                "v0.1 adjusted_coupling must be entirely NA",
                code="coupling_not_available_v0_1",
                field="adjusted_coupling",
                remediation="Do not infer coupling from context/batch co-variation",
            )
        if (numeric_columns["coupling_weight"] != 0).any():
            raise ContractError(
                "v0.1 coupling_weight must be zero",
                code="coupling_not_available_v0_1",
                field="coupling_weight",
                remediation="Exclude unvalidated coupling from the evidence score",
            )
        expected_coupling_status = SenderCouplingStatus.NOT_ESTIMABLE_V0_1.value
        if set(table["coupling_status"]) not in (set(), {expected_coupling_status}):
            raise ContractError(
                "v0.1 coupling status must be not_estimable_v0_1",
                code="coupling_not_available_v0_1",
                field="coupling_status",
                remediation="Preserve the explicit v0.1 coupling sentinel",
            )
        if set(table["coupling_reason_code"]) not in (set(), {_COUPLING_REASON}):
            raise ContractError(
                "v0.1 coupling requires its disabled reason code",
                code="coupling_not_available_v0_1",
                field="coupling_reason_code",
                remediation="Record why coupling is not estimable",
            )
        if set(table["assignment_mode"]) not in (set(), {_ASSIGNMENT_MODE}):
            raise ContractError(
                "v0.1 sender assignment must be exploratory in-sample",
                code="invalid_sender_assignment",
                field="assignment_mode",
                remediation="Do not label the descriptive producer as cross-fitted",
            )
        allowed_status = {status.value for status in SenderAssignmentStatus}
        if not set(table["status"]).issubset(allowed_status):
            raise ContractError(
                "sender assignment status is not recognized",
                code="invalid_sender_assignment",
                field="status",
                remediation="Use a released SenderAssignmentStatus value",
            )
        for row in table.itertuples(index=False):
            status = SenderAssignmentStatus(cast(str, row.status))
            reason_missing = pd.isna(row.reason_code)
            evidence_missing = pd.isna(row.evidence_score)
            if status is SenderAssignmentStatus.OK and not reason_missing:
                raise ContractError(
                    "status=ok must not carry a reason_code",
                    code="invalid_sender_assignment",
                    field="reason_code",
                    remediation="Reserve reasons for degraded assignment rows",
                )
            if status is not SenderAssignmentStatus.OK and reason_missing:
                raise ContractError(
                    "degraded sender status requires an explicit reason_code",
                    code="invalid_sender_assignment",
                    field="reason_code",
                    remediation="Record missing, partial, or low-support evidence",
                )
            if (status is SenderAssignmentStatus.MISSING_EVIDENCE) != evidence_missing:
                raise ContractError(
                    "missing_evidence status must agree with evidence_score=NA",
                    code="invalid_sender_assignment",
                    field="evidence_score",
                    remediation="Keep missing evidence distinct from observed zero",
                )
        for column in ("n_subjects", "n_group_subjects"):
            numeric = pd.to_numeric(table[column], errors="coerce")
            if numeric.isna().any() or any(
                not float(value).is_integer() or float(value) < 0
                for value in numeric.tolist()
            ):
                raise ContractError(
                    f"{column} must contain non-negative integer support",
                    code="invalid_sender_assignment",
                    field=column,
                    remediation="Count unique subject IDs at the declared grain",
                )
        grouped = table.groupby(
            list(ASSIGNMENT_GROUP_COLUMNS), sort=False, observed=True
        )
        for _, group in grouped:
            raw_weights = group["assignment_weight"]
            if raw_weights.isna().all():
                if not (
                    group["status"] == SenderAssignmentStatus.MISSING_EVIDENCE.value
                ).all() or group["normalized_entropy"].notna().any():
                    raise ContractError(
                        "all-missing sender evidence requires NA weights and entropy",
                        code="invalid_sender_normalization",
                        field="assignment_weight",
                        remediation="Do not emit fallback weights without evidence",
                    )
                continue
            if raw_weights.isna().any() or group["normalized_entropy"].isna().any():
                raise ContractError(
                    "partially missing sender weights are not a valid assignment",
                    code="invalid_sender_normalization",
                    field="assignment_weight",
                    remediation="Normalize observed evidence or mark the group missing",
                )
            weights = [float(value) for value in raw_weights]
            if not math.isclose(sum(weights), 1.0, rel_tol=1e-10, abs_tol=1e-12):
                raise ContractError(
                    "sender assignment weights must sum to one within every group",
                    code="invalid_sender_normalization",
                    field="assignment_weight",
                    remediation="Apply one stable softmax across all candidate senders",
                )
            candidate_count = len(weights)
            entropy = 0.0
            if candidate_count > 1:
                entropy = -sum(
                    weight * math.log(weight) for weight in weights if weight > 0
                ) / math.log(candidate_count)
            observed_entropy = [float(value) for value in group["normalized_entropy"]]
            if any(
                not math.isclose(value, entropy, rel_tol=1e-10, abs_tol=1e-12)
                for value in observed_entropy
            ):
                raise ContractError(
                    "normalized_entropy must match the group's assignment weights",
                    code="invalid_sender_entropy",
                    field="normalized_entropy",
                    remediation="Use normalized Shannon entropy of assignment weights",
                )
            group_support = {int(value) for value in group["n_group_subjects"]}
            if len(group_support) != 1 or any(
                int(cast(Any, row.n_subjects)) > int(cast(Any, row.n_group_subjects))
                for row in group.itertuples(index=False)
            ):
                raise ContractError(
                    "subject support is inconsistent within a sender group",
                    code="invalid_sender_assignment",
                    field="n_group_subjects",
                    remediation=(
                        "Use the group's unique subject universe as denominator"
                    ),
                )
        table = table.loc[:, list(SENDER_ASSIGNMENT_COLUMNS)].sort_values(
            [*ASSIGNMENT_GROUP_COLUMNS, "sender"],
            kind="stable",
            ignore_index=True,
        )
        object.__setattr__(self, "table", table)

    @property
    def inference_eligible(self) -> bool:
        """Sender evidence is descriptive and never a formal test in v0.1."""

        return False

    @property
    def causal_interpretation(self) -> str:
        """State the interpretation boundary for downstream consumers."""

        return "evidence_based_non_causal"

    @property
    def coupling_estimable(self) -> bool:
        """Adjusted coupling is deliberately disabled in v0.1."""

        return False

    def query(
        self,
        *,
        context_id: str | None = None,
        receiver: str | None = None,
        interaction_id: str | None = None,
        sender: str | None = None,
    ) -> pd.DataFrame:
        """Return a defensive copy filtered at sender-assignment grain."""

        selected = pd.Series(True, index=self.table.index, dtype=bool)
        for column, value in (
            ("context_id", context_id),
            ("receiver", receiver),
            ("interaction_id", interaction_id),
            ("sender", sender),
        ):
            if value is not None:
                selected &= self.table[column] == value
        return self.table.loc[selected].copy(deep=True).reset_index(drop=True)
