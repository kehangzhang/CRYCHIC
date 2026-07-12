"""Deterministic v0.1 sender evidence aggregation and soft assignment."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from crychic.core import ContractError

from .contracts import (
    ASSIGNMENT_GROUP_COLUMNS,
    SENDER_ASSIGNMENT_COLUMNS,
    SenderAssignment,
    SenderAssignmentStatus,
    SenderCouplingStatus,
    SenderEvidenceParameters,
    sender_assignment_id,
)

_INPUT_COLUMNS = {
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "ligand_availability",
}
_INPUT_KEY = (
    "sample_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_FORBIDDEN_ATTRIBUTION_COLUMNS = {
    "attribution_coefficient",
    "coefficient",
    "driver_coefficient",
    "family_coefficient",
    "selection_frequency",
    "specificity_support",
}
_COUPLING_REASON = "v0_1_adjusted_coupling_disabled"
_ASSIGNMENT_MODE = "exploratory_in_sample"


def _validate_input(sample_availability: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sample_availability, pd.DataFrame):
        raise TypeError("sample_availability must be a pandas DataFrame")
    missing = _INPUT_COLUMNS.difference(sample_availability.columns)
    if missing:
        raise ContractError(
            f"sample availability is missing columns: {sorted(missing)}",
            code="invalid_sender_input",
            field="columns",
            remediation="Supply sample-level ligand availability components",
        )
    forbidden = _FORBIDDEN_ATTRIBUTION_COLUMNS.intersection(sample_availability.columns)
    if forbidden:
        raise ContractError(
            "sender assignment must not consume attribution fields: "
            f"{sorted(forbidden)}",
            code="forbidden_sender_attribution",
            field="columns",
            remediation="Pass availability components without receiver coefficients",
        )
    table = sample_availability.copy(deep=True)
    identifier_columns = (
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    )
    for column in identifier_columns:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in table[column].tolist()
        ):
            raise ContractError(
                f"{column} must contain non-empty string identifiers",
                code="invalid_sender_input",
                field=column,
                remediation="Use stable sample, subject, context, and entity IDs",
            )
    if table.duplicated(list(_INPUT_KEY)).any():
        raise ContractError(
            "sample-level sender availability keys must be unique",
            code="duplicate_sender_input",
            field="sample_id",
            remediation="Emit one sender availability row per sample and interaction",
        )
    sample_subject_count = table.groupby("sample_id", observed=True)[
        "subject_id"
    ].nunique()
    if (sample_subject_count > 1).any():
        raise ContractError(
            "each sample_id must map to one subject_id",
            code="invalid_sender_input",
            field="subject_id",
            remediation="Restore immutable sample-to-subject linkage",
        )
    numeric = pd.to_numeric(table["ligand_availability"], errors="coerce")
    invalid_type = table["ligand_availability"].notna() & numeric.isna()
    present = [float(value) for value in numeric.dropna().tolist()]
    if invalid_type.any() or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in present
    ):
        raise ContractError(
            "ligand_availability must contain unit-interval values or NA",
            code="invalid_sender_input",
            field="ligand_availability",
            remediation="Use the sample-level availability producer scale",
        )
    table["ligand_availability"] = numeric.astype(float)
    return table


def _softmax(
    scores: list[float | None], *, temperature: float
) -> list[float | None]:
    observed = {index: value for index, value in enumerate(scores) if value is not None}
    if not observed:
        return [None] * len(scores)
    maximum = max(observed.values())
    exponentials = {
        index: math.exp((value - maximum) / temperature)
        for index, value in observed.items()
    }
    denominator = sum(exponentials.values())
    return [exponentials.get(index, 0.0) / denominator for index in range(len(scores))]


def _normalized_entropy(weights: list[float | None]) -> float | None:
    if all(weight is None for weight in weights):
        return None
    if any(weight is None for weight in weights):
        raise ValueError("sender weights must be either all missing or all numeric")
    numeric = [float(weight) for weight in weights if weight is not None]
    if len(numeric) <= 1:
        return 0.0
    entropy = float(
        -sum(weight * math.log(weight) for weight in numeric if weight > 0)
        / math.log(len(numeric))
    )
    return min(1.0, max(0.0, entropy))


def _evidence_score(
    *,
    ligand_availability: float,
    cell_type_specificity: float,
    subject_prevalence: float,
    parameters: SenderEvidenceParameters,
) -> float:
    values = {
        "ligand_availability": ligand_availability,
        "cell_type_specificity": cell_type_specificity,
        "subject_prevalence": subject_prevalence,
    }
    active = {
        component: parameters.component_weights[component]
        for component in values
        if parameters.component_weights[component] > 0
    }
    denominator = sum(active.values())
    return float(
        sum(active[component] * values[component] for component in active) / denominator
    )


def assign_senders(
    sample_availability: pd.DataFrame,
    parameters: SenderEvidenceParameters | None = None,
) -> SenderAssignment:
    """Build non-causal sender evidence from sample-level ligand availability.

    Ligand evidence is averaged within subject before cross-subject aggregation.
    Specificity is the sender's share of group-level ligand availability, and
    prevalence is the fraction of group subjects above the frozen threshold.
    No receiver outcome, attribution coefficient, or hypothesis-test quantity
    is consumed. Adjusted coupling is therefore explicitly unavailable in v0.1.
    """

    resolved = parameters or SenderEvidenceParameters()
    if not isinstance(resolved, SenderEvidenceParameters):
        raise TypeError("parameters must be SenderEvidenceParameters or None")
    table = _validate_input(sample_availability)
    if table.empty:
        return SenderAssignment(
            pd.DataFrame(columns=SENDER_ASSIGNMENT_COLUMNS), resolved
        )

    output_rows: list[dict[str, Any]] = []
    grouped = table.groupby(list(ASSIGNMENT_GROUP_COLUMNS), sort=True, observed=True)
    for group_key, group in grouped:
        context_id, receiver, interaction_id = group_key
        candidates = sorted(str(value) for value in group["sender"].unique())
        n_group_subjects = int(group["subject_id"].nunique())
        candidate_summaries: list[dict[str, Any]] = []
        for sender in candidates:
            sender_rows = group.loc[group["sender"] == sender]
            subject_values = sender_rows.groupby("subject_id", observed=True)[
                "ligand_availability"
            ].mean()
            observed = subject_values.dropna()
            n_subjects = len(observed)
            if n_subjects == 0:
                ligand_availability: float | None = None
                subject_prevalence: float | None = None
            else:
                ligand_availability = float(observed.mean())
                subject_prevalence = float(
                    (observed > resolved.prevalence_threshold).sum() / n_group_subjects
                )
            candidate_summaries.append(
                {
                    "sender": sender,
                    "ligand_availability": ligand_availability,
                    "subject_prevalence": subject_prevalence,
                    "n_subjects": n_subjects,
                }
            )

        specificity_denominator = sum(
            float(summary["ligand_availability"])
            for summary in candidate_summaries
            if summary["ligand_availability"] is not None
        )
        scores: list[float | None] = []
        for summary in candidate_summaries:
            ligand_availability = summary["ligand_availability"]
            prevalence = summary["subject_prevalence"]
            if ligand_availability is None or prevalence is None:
                specificity: float | None = None
                score: float | None = None
            else:
                specificity = (
                    float(ligand_availability) / specificity_denominator
                    if specificity_denominator > 0
                    else 0.0
                )
                score = _evidence_score(
                    ligand_availability=float(ligand_availability),
                    cell_type_specificity=specificity,
                    subject_prevalence=float(prevalence),
                    parameters=resolved,
                )
            summary["cell_type_specificity"] = specificity
            summary["evidence_score"] = score
            scores.append(score)

        weights = _softmax(scores, temperature=resolved.softmax_temperature)
        entropy = _normalized_entropy(weights)
        group_has_missing = any(score is None for score in scores)
        group_all_missing = all(score is None for score in scores)
        for summary, assignment_weight in zip(
            candidate_summaries, weights, strict=True
        ):
            score = summary["evidence_score"]
            n_subjects = int(summary["n_subjects"])
            if score is None:
                status = SenderAssignmentStatus.MISSING_EVIDENCE
                reason_code: str | None = (
                    "all_candidate_evidence_missing"
                    if group_all_missing
                    else "missing_ligand_availability"
                )
            elif n_subjects < resolved.min_subjects:
                status = SenderAssignmentStatus.LOW_SUPPORT
                reason_code = "insufficient_subject_support"
            elif group_has_missing:
                status = SenderAssignmentStatus.PARTIAL_EVIDENCE
                reason_code = "partial_candidate_evidence"
            else:
                status = SenderAssignmentStatus.OK
                reason_code = None
            sender = str(summary["sender"])
            output_rows.append(
                {
                    "assignment_id": sender_assignment_id(
                        functional_id=resolved.assignment_functional_id,
                        context_id=str(context_id),
                        receiver=str(receiver),
                        interaction_id=str(interaction_id),
                        sender=sender,
                        schema_version=resolved.schema_version,
                    ),
                    "assignment_functional_id": resolved.assignment_functional_id,
                    "context_id": str(context_id),
                    "receiver": str(receiver),
                    "interaction_id": str(interaction_id),
                    "sender": sender,
                    "ligand_availability": summary["ligand_availability"],
                    "cell_type_specificity": summary["cell_type_specificity"],
                    "subject_prevalence": summary["subject_prevalence"],
                    "adjusted_coupling": None,
                    "coupling_weight": 0.0,
                    "evidence_score": score,
                    "assignment_weight": assignment_weight,
                    "normalized_entropy": entropy,
                    "n_subjects": n_subjects,
                    "n_group_subjects": n_group_subjects,
                    "status": status.value,
                    "reason_code": reason_code,
                    "coupling_status": (SenderCouplingStatus.NOT_ESTIMABLE_V0_1.value),
                    "coupling_reason_code": _COUPLING_REASON,
                    "assignment_mode": _ASSIGNMENT_MODE,
                }
            )
    output = pd.DataFrame(output_rows, columns=SENDER_ASSIGNMENT_COLUMNS)
    return SenderAssignment(output, resolved)
