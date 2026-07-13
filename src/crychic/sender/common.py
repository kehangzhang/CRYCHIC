"""Train/apply primitives for a contrast-common sender functional."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any, cast

import pandas as pd

from crychic.core import ContractError, stable_id
from crychic.design import ContrastSpec

from .contracts import (
    COMMON_SENDER_APPLICATION_COLUMNS,
    COMMON_SENDER_GROUP_COLUMNS,
    CommonSenderApplication,
    CommonSenderApplicationStatus,
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    SenderPrevalencePrior,
    SenderPrevalenceStatus,
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
_FORBIDDEN_COLUMNS = {
    "attribution_coefficient",
    "coefficient",
    "driver_coefficient",
    "family_coefficient",
    "receiver_activity",
    "response",
    "selection_frequency",
    "specificity_support",
}
_APPLICATION_MODE = "frozen_contrast_common_partial_not_oof"


def _validated_availability(table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("sample_availability must be a pandas DataFrame")
    missing = _INPUT_COLUMNS.difference(table.columns)
    if missing:
        raise ContractError(
            f"sample availability is missing columns: {sorted(missing)}",
            code="invalid_common_sender_input",
            field="columns",
            remediation="Supply sample-local ligand availability rows",
        )
    forbidden = _FORBIDDEN_COLUMNS.intersection(table.columns)
    if forbidden:
        raise ContractError(
            "common sender functional must not consume receiver outcomes: "
            f"{sorted(forbidden)}",
            code="forbidden_common_sender_outcome",
            field="columns",
            remediation="Use ligand availability without attribution or response data",
        )
    result = table.copy(deep=True)
    for column in (
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    ):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in result[column].astype(object).tolist()
        ):
            raise ContractError(
                f"{column} must contain non-empty string identifiers",
                code="invalid_common_sender_input",
                field=column,
                remediation="Use stable sample, context, and entity IDs",
            )
        result[column] = result[column].astype(str)
    if result.duplicated(list(_INPUT_KEY)).any():
        raise ContractError(
            "sample-local sender availability keys must be unique",
            code="invalid_common_sender_input",
            field="sample_id",
            remediation="Emit one sender row per sample and interaction",
        )
    if result.groupby("sample_id", observed=True)["subject_id"].nunique().gt(1).any():
        raise ContractError(
            "each sample_id must map to one subject_id",
            code="invalid_common_sender_input",
            field="subject_id",
            remediation="Restore immutable sample-to-subject linkage",
        )
    numeric = pd.to_numeric(result["ligand_availability"], errors="coerce")
    invalid = result["ligand_availability"].notna() & numeric.isna()
    present = numeric.dropna()
    if invalid.any() or (
        not present.empty
        and (
            (present < 0).any()
            or (present > 1).any()
            or not present.map(math.isfinite).all()
        )
    ):
        raise ContractError(
            "ligand_availability must contain unit-interval values or NA",
            code="invalid_common_sender_input",
            field="ligand_availability",
            remediation="Use the sample-level availability producer scale",
        )
    result["ligand_availability"] = numeric.astype(float)
    return result


def fit_contrast_common_sender_functional(
    training_availability: pd.DataFrame,
    *,
    contrast: ContrastSpec,
    context_keys: Sequence[str],
    filter_universe_id: str,
    parameters: ContrastCommonSenderParameters | None = None,
) -> ContrastCommonSenderFunctional:
    """Fit one sender universe and prevalence prior across training contexts."""

    resolved = parameters or ContrastCommonSenderParameters()
    if not isinstance(resolved, ContrastCommonSenderParameters):
        raise TypeError("parameters must be ContrastCommonSenderParameters or None")
    if not isinstance(filter_universe_id, str) or not filter_universe_id:
        raise ValueError("filter_universe_id must be a non-empty string")
    if not isinstance(contrast, ContrastSpec):
        raise TypeError("contrast must be a ContrastSpec")
    if not contrast.estimable or len(contrast.weights) < 2:
        raise ValueError(
            "common sender fitting requires an estimable multi-context contrast"
        )
    keys = tuple(context_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("context_keys must contain non-empty identifiers")
    table = _validated_availability(training_availability)
    if table.empty:
        raise ValueError("common sender fitting requires training availability rows")
    node_to_context_id: dict[Hashable, str] = {}
    for row in table.itertuples(index=False):
        if len(keys) == 1:
            node: Hashable = getattr(row, keys[0])
        else:
            node = tuple((key, getattr(row, key)) for key in keys)
        context_identifier = str(row.context_id)
        previous = node_to_context_id.setdefault(node, context_identifier)
        if previous != context_identifier:
            raise ValueError("context node maps to multiple context IDs")
    missing_nodes = set(contrast.weights).difference(node_to_context_id)
    if missing_nodes:
        raise ValueError("contrast contexts are absent from training availability")
    contrast_weights = tuple(
        (node_to_context_id[node], float(weight))
        for node, weight in contrast.weights.items()
    )
    contexts = tuple(sorted(context_id for context_id, _ in contrast_weights))
    table = table.loc[table["context_id"].isin(contexts)].copy()
    subjects = tuple(sorted(table["subject_id"].unique()))
    denominator = len(subjects)
    priors: list[SenderPrevalencePrior] = []
    for (receiver, interaction_id, sender), group in table.groupby(
        ["receiver", "interaction_id", "sender"], observed=True, sort=True
    ):
        subject_values = group.groupby("subject_id", observed=True)[
            "ligand_availability"
        ].mean()
        observed = subject_values.dropna()
        n_subjects = len(observed)
        if n_subjects == 0:
            prior = None
            status = SenderPrevalenceStatus.MISSING_EVIDENCE
            reason = "training_ligand_evidence_missing"
        elif n_subjects < resolved.min_subjects:
            prior = None
            status = SenderPrevalenceStatus.LOW_SUPPORT
            reason = "insufficient_training_subject_support"
        else:
            prior = float(
                (observed > resolved.prevalence_threshold).sum() / denominator
            )
            status = SenderPrevalenceStatus.SUPPORTED
            reason = None
        priors.append(
            SenderPrevalencePrior(
                receiver=str(receiver),
                interaction_id=str(interaction_id),
                sender=str(sender),
                prevalence_prior=prior,
                n_subjects=n_subjects,
                status=status,
                reason_code=reason,
            )
        )
    return ContrastCommonSenderFunctional(
        contrast=contrast,
        contrast_context_ids=tuple(
            (node, node_to_context_id[node]) for node in contrast.weights
        ),
        training_subject_ids=subjects,
        filter_universe_id=filter_universe_id,
        parameters=resolved,
        candidate_priors=tuple(priors),
    )


def _softmax(values: list[float | None], *, temperature: float) -> list[float | None]:
    observed = {index: value for index, value in enumerate(values) if value is not None}
    if not observed:
        return [None] * len(values)
    maximum = max(observed.values())
    exponentials = {
        index: math.exp((value - maximum) / temperature)
        for index, value in observed.items()
    }
    denominator = sum(exponentials.values())
    return [exponentials.get(index, 0.0) / denominator for index in range(len(values))]


def _entropy(weights: list[float | None]) -> float | None:
    if all(weight is None for weight in weights):
        return None
    numeric = [float(weight) for weight in weights if weight is not None]
    if len(numeric) <= 1:
        return 0.0
    return float(
        -sum(weight * math.log(weight) for weight in numeric if weight > 0)
        / math.log(len(numeric))
    )


def apply_contrast_common_sender_functional(
    functional: ContrastCommonSenderFunctional,
    sample_availability: pd.DataFrame,
) -> CommonSenderApplication:
    """Apply frozen priors to sample-local ligand evidence without refitting."""

    if not isinstance(functional, ContrastCommonSenderFunctional):
        raise TypeError("functional must be ContrastCommonSenderFunctional")
    table = _validated_availability(sample_availability)
    table = table.loc[table["context_id"].isin(functional.context_ids)].copy()
    if table.empty:
        return CommonSenderApplication(
            pd.DataFrame(columns=COMMON_SENDER_APPLICATION_COLUMNS), functional
        )
    priors_by_group: dict[tuple[str, str], tuple[SenderPrevalencePrior, ...]] = {}
    for prior in functional.candidate_priors:
        priors_by_group.setdefault((prior.receiver, prior.interaction_id), ())
        priors_by_group[(prior.receiver, prior.interaction_id)] = (
            *priors_by_group[(prior.receiver, prior.interaction_id)],
            prior,
        )
    output: list[dict[str, Any]] = []
    group_columns = [*COMMON_SENDER_GROUP_COLUMNS]
    for group_key, group in table.groupby(group_columns, observed=True, sort=True):
        sample_id, subject_id, context_id, receiver, interaction_id = group_key
        priors = priors_by_group.get((str(receiver), str(interaction_id)))
        if priors is None:
            continue
        local = {
            str(row.sender): (
                None
                if pd.isna(row.ligand_availability)
                else float(cast(Any, row.ligand_availability))
            )
            for row in group.itertuples(index=False)
            if str(row.sender) in {prior.sender for prior in priors}
        }
        evidence: list[float | None] = []
        for prior in priors:
            ligand = local.get(prior.sender)
            evidence.append(
                None
                if ligand is None or prior.prevalence_prior is None
                else ligand * prior.prevalence_prior
            )
        weights = _softmax(
            evidence,
            temperature=functional.parameters.softmax_temperature,
        )
        if any(value is None for value in evidence):
            weights = [None] * len(evidence)
        entropy = _entropy(weights)
        group_all_missing = all(value is None for value in evidence)
        group_has_missing = any(value is None for value in evidence)
        for prior, raw_evidence, weight in zip(priors, evidence, weights, strict=True):
            ligand = local.get(prior.sender)
            reason: str | None
            if group_all_missing:
                status = CommonSenderApplicationStatus.NOT_ESTIMABLE
                reason = "all_candidate_evidence_missing"
            elif group_has_missing:
                status = CommonSenderApplicationStatus.NOT_ESTIMABLE
                reason = "incomplete_candidate_evidence"
            else:
                status = CommonSenderApplicationStatus.OK
                reason = None
            output.append(
                {
                    "sender_application_id": stable_id(
                        "common_sender_application",
                        {
                            "context_id": str(context_id),
                            "interaction_id": str(interaction_id),
                            "receiver": str(receiver),
                            "sample_id": str(sample_id),
                            "subject_id": str(subject_id),
                            "sender": prior.sender,
                            "sender_functional_id": functional.sender_functional_id,
                        },
                    ),
                    "sender_functional_id": functional.sender_functional_id,
                    "sample_id": str(sample_id),
                    "subject_id": str(subject_id),
                    "context_id": str(context_id),
                    "receiver": str(receiver),
                    "interaction_id": str(interaction_id),
                    "sender": prior.sender,
                    "ligand_availability": ligand,
                    "training_prevalence_prior": prior.prevalence_prior,
                    "raw_sender_evidence": raw_evidence,
                    "assignment_weight": weight,
                    "normalized_entropy": entropy,
                    "training_n_subjects": prior.n_subjects,
                    "status": status.value,
                    "reason_code": reason,
                    "assignment_mode": _APPLICATION_MODE,
                }
            )
    return CommonSenderApplication(
        pd.DataFrame(output, columns=COMMON_SENDER_APPLICATION_COLUMNS), functional
    )


_UNRESOLVED_COLUMNS = {
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "interaction_id",
    "mode",
    "sender_unresolved_strength",
}


def allocate_sender_resolved_strength(
    application: CommonSenderApplication,
    unresolved_strengths: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate, but never change, sender-unresolved LR strength."""

    if not isinstance(application, CommonSenderApplication):
        raise TypeError("application must be CommonSenderApplication")
    if not isinstance(unresolved_strengths, pd.DataFrame):
        raise TypeError("unresolved_strengths must be a pandas DataFrame")
    missing = _UNRESOLVED_COLUMNS.difference(unresolved_strengths.columns)
    if missing:
        raise ValueError(f"unresolved strengths are missing: {sorted(missing)}")
    unresolved = unresolved_strengths.loc[:, sorted(_UNRESOLVED_COLUMNS)].copy()
    keys = [*COMMON_SENDER_GROUP_COLUMNS, "mode"]
    if unresolved.duplicated(keys).any():
        raise ValueError("sender-unresolved strength keys must be unique")
    numeric = pd.to_numeric(unresolved["sender_unresolved_strength"], errors="coerce")
    invalid = unresolved["sender_unresolved_strength"].notna() & numeric.isna()
    present = numeric.dropna()
    if invalid.any() or (
        not present.empty
        and (
            (present < 0).any()
            or (present > 1).any()
            or not present.map(math.isfinite).all()
        )
    ):
        raise ValueError("sender_unresolved_strength must lie in [0, 1] or be NA")
    unresolved["sender_unresolved_strength"] = numeric.astype(float)
    assignment_groups = application.table.loc[
        :, list(COMMON_SENDER_GROUP_COLUMNS)
    ].drop_duplicates()
    unresolved_groups = unresolved.loc[
        :, list(COMMON_SENDER_GROUP_COLUMNS)
    ].drop_duplicates()
    if (
        not assignment_groups.merge(
            unresolved_groups,
            how="outer",
            on=list(COMMON_SENDER_GROUP_COLUMNS),
            indicator=True,
        )["_merge"]
        .eq("both")
        .all()
    ):
        raise ValueError("unresolved strengths do not cover sender application groups")
    merged = application.table.merge(
        unresolved,
        how="inner",
        on=list(COMMON_SENDER_GROUP_COLUMNS),
        validate="many_to_many",
        sort=False,
    )
    resolved_values: list[float | None] = []
    statuses: list[str] = []
    reasons: list[str | None] = []
    for row in merged.itertuples(index=False):
        unresolved_value = cast(Any, row.sender_unresolved_strength)
        assignment_weight = cast(Any, row.assignment_weight)
        if pd.isna(unresolved_value):
            resolved_values.append(None)
            statuses.append("not_estimable")
            reasons.append("sender_unresolved_strength_missing")
        elif pd.isna(assignment_weight):
            resolved_values.append(None)
            statuses.append("not_estimable")
            reasons.append("sender_assignment_not_estimable")
        else:
            resolved_values.append(float(unresolved_value) * float(assignment_weight))
            statuses.append("ok")
            reasons.append(None)
    result = merged.loc[
        :,
        [
            "sender_functional_id",
            *COMMON_SENDER_GROUP_COLUMNS,
            "mode",
            "sender",
            "sender_unresolved_strength",
            "assignment_weight",
        ],
    ].copy()
    result["sender_resolved_strength"] = resolved_values
    result["status"] = statuses
    result["reason_code"] = reasons
    group_keys = [*COMMON_SENDER_GROUP_COLUMNS, "mode"]
    for _, group in result.groupby(group_keys, observed=True, sort=False):
        unresolved_value = group["sender_unresolved_strength"].iloc[0]
        resolved = group["sender_resolved_strength"]
        if (
            pd.notna(unresolved_value)
            and resolved.notna().all()
            and not math.isclose(
                float(resolved.sum()),
                float(unresolved_value),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("sender allocation failed exact conservation")
    return result.sort_values([*group_keys, "sender"], kind="stable", ignore_index=True)


__all__ = [
    "allocate_sender_resolved_strength",
    "apply_contrast_common_sender_functional",
    "fit_contrast_common_sender_functional",
]
