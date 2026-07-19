"""Train/apply primitives for a contrast-common sender functional."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from scipy.stats import t as student_t

from crychic.availability import FrozenInteractionUniverse
from crychic.core import ContractError, canonical_json, stable_id
from crychic.core._validation import validation_scope
from crychic.design import ContrastSpec, canonical_context, plain_context_value

from .contracts import (
    _SENDER_CONTRAST_MULTIPLICITY_METHOD,
    _SENDER_CONTRAST_MULTIPLICITY_SCOPE,
    _SENDER_CONTRAST_SUPPORT_POLICY,
    _SENDER_CONTRAST_SUPPORT_POLICY_INDEPENDENT,
    _SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN,
    _SENDER_FUNCTIONAL_PRODUCER_TOKEN,
    COMMON_SENDER_APPLICATION_COLUMNS,
    COMMON_SENDER_GROUP_COLUMNS,
    CommonSenderApplication,
    CommonSenderApplicationStatus,
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    InteractionLigandContrastGate,
    InteractionLigandContrastSupport,
    SenderContrastSupportStatus,
    SenderPrevalencePrior,
    SenderPrevalenceStatus,
    _canonical_contrast_weights,
    _holm_step_down_adjustments,
    _sender_contrast_family_id,
    _sender_contrast_familywise_alpha,
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
_TRAINING_DIGEST_COLUMNS = (
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "ligand_availability",
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


@dataclass(frozen=True, slots=True)
class _RawInteractionContrast:
    receiver: str
    interaction_id: str
    complete_subject_ids: tuple[str, ...]
    subject_effects: tuple[float, ...]
    unit_context_ids: tuple[str, ...]
    degrees_of_freedom: float | int | None
    mean_effect: float | None
    sample_standard_error: float | None
    critical_value: float | None
    lower_confidence_bound: float | None
    raw_one_sided_p_value: float | None


def _normalized_frozen_interaction_ids(
    frozen_interaction_ids: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(frozen_interaction_ids, (str, bytes)):
        raise ContractError(
            "frozen_interaction_ids must be a sequence of identifiers",
            code="invalid_common_sender_input",
            field="frozen_interaction_ids",
            remediation="Pass the exact outer-training interaction universe",
        )
    values = tuple(frozen_interaction_ids)
    if not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ContractError(
            "frozen_interaction_ids must contain non-empty identifiers",
            code="invalid_common_sender_input",
            field="frozen_interaction_ids",
            remediation="Pass the exact outer-training interaction universe",
        )
    normalized = tuple(sorted(value.strip() for value in values))
    if len(normalized) != len(set(normalized)):
        raise ContractError(
            "frozen_interaction_ids must be unique",
            code="invalid_common_sender_input",
            field="frozen_interaction_ids",
            remediation="Deduplicate the frozen interaction manifest",
        )
    return normalized


def _frozen_candidate_sender_manifest(
    table: pd.DataFrame,
    *,
    frozen_interaction_ids: tuple[str, ...],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Expand the receiver-wise frozen interaction and candidate-sender grid."""

    observed_interactions = set(table["interaction_id"].astype(str))
    unknown = observed_interactions.difference(frozen_interaction_ids)
    if unknown:
        raise ContractError(
            "training availability contains interactions outside the frozen universe",
            code="invalid_common_sender_input",
            field="interaction_id",
            remediation="Apply the exact training-fold interaction universe",
        )
    candidate_rows = table.loc[
        :, ["receiver", "interaction_id", "sender"]
    ].drop_duplicates()
    senders_by_receiver: dict[str, set[str]] = {}
    senders_by_candidate: dict[tuple[str, str], set[str]] = {}
    for receiver, interaction_id, sender in candidate_rows.itertuples(
        index=False, name=None
    ):
        receiver_id = str(receiver)
        interaction = str(interaction_id)
        sender_id = str(sender)
        senders_by_receiver.setdefault(receiver_id, set()).add(sender_id)
        senders_by_candidate.setdefault((receiver_id, interaction), set()).add(
            sender_id
        )
    manifest: list[tuple[str, str, tuple[str, ...]]] = []
    for receiver in sorted(senders_by_receiver):
        receiver_senders = tuple(sorted(senders_by_receiver[receiver]))
        if not receiver_senders:
            raise ContractError(
                "every frozen receiver must have at least one candidate sender",
                code="invalid_common_sender_input",
                field="sender",
                remediation="Freeze candidates from the outer-training availability",
            )
        for interaction_id in frozen_interaction_ids:
            local_senders = tuple(
                sorted(senders_by_candidate.get((receiver, interaction_id), ()))
            )
            manifest.append(
                (
                    receiver,
                    interaction_id,
                    local_senders or receiver_senders,
                )
            )
    return tuple(manifest)


def freeze_common_sender_candidate_manifest(
    training_availability: pd.DataFrame,
    *,
    frozen_interaction_ids: Sequence[str],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Freeze receiver, interaction, and candidate-sender IDs before fitting."""

    table = _validated_availability(training_availability)
    if table.empty:
        raise ValueError("candidate freezing requires training availability rows")
    interactions = _normalized_frozen_interaction_ids(frozen_interaction_ids)
    return _frozen_candidate_sender_manifest(
        table,
        frozen_interaction_ids=interactions,
    )


def _validated_candidate_sender_manifest(
    manifest: Sequence[Sequence[object]],
    *,
    frozen_interaction_ids: tuple[str, ...],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    normalized: list[tuple[str, str, tuple[str, ...]]] = []
    for raw_entry in tuple(manifest):
        if (
            not isinstance(raw_entry, Sequence)
            or isinstance(raw_entry, (str, bytes))
            or len(raw_entry) != 3
        ):
            raise ContractError(
                "frozen candidate sender entries must have three fields",
                code="invalid_common_sender_input",
                field="frozen_candidate_sender_manifest",
                remediation="Freeze receiver, interaction, and sender identifiers",
            )
        receiver, interaction_id, raw_senders = raw_entry
        if (
            not isinstance(receiver, str)
            or not receiver.strip()
            or not isinstance(interaction_id, str)
            or not interaction_id.strip()
        ):
            raise ContractError(
                "frozen candidate sender identifiers must be non-empty",
                code="invalid_common_sender_input",
                field="frozen_candidate_sender_manifest",
                remediation="Use stable receiver and interaction identifiers",
            )
        if not isinstance(raw_senders, Sequence) or isinstance(
            raw_senders, (str, bytes)
        ):
            raise ContractError(
                "frozen candidate senders must be an identifier sequence",
                code="invalid_common_sender_input",
                field="frozen_candidate_sender_manifest",
                remediation="Freeze candidates from complete training availability",
            )
        senders = tuple(raw_senders)
        if not senders or any(
            not isinstance(sender, str) or not sender.strip() for sender in senders
        ):
            raise ContractError(
                "every frozen interaction must contain candidate senders",
                code="invalid_common_sender_input",
                field="frozen_candidate_sender_manifest",
                remediation="Freeze candidates from complete training availability",
            )
        normalized_senders = tuple(sorted(sender.strip() for sender in senders))
        if len(normalized_senders) != len(set(normalized_senders)):
            raise ContractError(
                "frozen candidate senders must be unique per interaction",
                code="invalid_common_sender_input",
                field="frozen_candidate_sender_manifest",
                remediation="Deduplicate the frozen candidate universe",
            )
        normalized.append(
            (receiver.strip(), interaction_id.strip(), normalized_senders)
        )
    result = tuple(sorted(normalized))
    keys = [(receiver, interaction) for receiver, interaction, _ in result]
    if not result or len(keys) != len(set(keys)):
        raise ContractError(
            "frozen candidate sender keys must be non-empty and unique",
            code="invalid_common_sender_input",
            field="frozen_candidate_sender_manifest",
            remediation="Emit one manifest row per receiver and interaction",
        )
    for receiver in sorted({receiver for receiver, _, _ in result}):
        receiver_interactions = tuple(
            interaction
            for candidate_receiver, interaction, _ in result
            if candidate_receiver == receiver
        )
        if receiver_interactions != frozen_interaction_ids:
            raise ContractError(
                "every receiver must bind the complete frozen interaction universe",
                code="invalid_common_sender_input",
                field="frozen_candidate_sender_manifest",
                remediation="Materialize zero-row interactions before fitting",
            )
    return result


def _training_availability_digest(
    table: pd.DataFrame,
    *,
    contrast_context_ids: tuple[tuple[Hashable, str], ...],
    contrast_weights: tuple[tuple[str, float], ...],
    filter_universe_id: str,
    frozen_interaction_ids: tuple[str, ...],
    candidate_sender_manifest: tuple[tuple[str, str, tuple[str, ...]], ...],
) -> str:
    """Digest the exact canonical training rows consumed by one contrast fit."""

    canonical = table.loc[:, list(_TRAINING_DIGEST_COLUMNS)].sort_values(
        list(_TRAINING_DIGEST_COLUMNS[:-1]), kind="stable", ignore_index=True
    )
    row_digest = hashlib.sha256()
    encoded_tokens: dict[str, bytes] = {}
    digest_buffer = bytearray()
    flush_bytes = 1 << 20

    def encoded_token(value: str) -> bytes:
        token = encoded_tokens.get(value)
        if token is None:
            token = canonical_json(value).encode("ascii")
            encoded_tokens[value] = token
        return token

    for row in canonical.itertuples(index=False, name=None):
        ligand = row[-1]
        ligand_token = (
            b"null"
            if pd.isna(ligand)
            else encoded_token(float(ligand).hex())
        )
        digest_buffer.extend(b"[")
        digest_buffer.extend(b",".join(encoded_token(value) for value in row[:-1]))
        digest_buffer.extend(b",")
        digest_buffer.extend(ligand_token)
        digest_buffer.extend(b"]\n")
        if len(digest_buffer) >= flush_bytes:
            row_digest.update(digest_buffer)
            digest_buffer.clear()
    row_digest.update(digest_buffer)
    canonical_weights = _canonical_contrast_weights(contrast_weights)
    payload: dict[str, object] = {
        "candidate_sender_manifest": [
            [receiver, interaction_id, list(sender_ids)]
            for receiver, interaction_id, sender_ids in candidate_sender_manifest
        ],
        "columns": list(_TRAINING_DIGEST_COLUMNS),
        "contrast_weights": [list(value) for value in canonical_weights],
        "filter_universe_id": filter_universe_id,
        "frozen_interaction_ids": list(frozen_interaction_ids),
        "row_count": len(canonical),
        "row_sha256": row_digest.hexdigest(),
    }
    if len(contrast_context_ids) > 2:
        payload["contrast_context_ids"] = [
            [node, context_id] for node, context_id in contrast_context_ids
        ]
    result: str = stable_id(
        "common_sender_training_availability",
        payload,
        schema_version="3",
        digest_length=64,
    )
    return result


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


def _fit_interaction_ligand_contrast_supports(
    table: pd.DataFrame,
    *,
    contrast_weights: tuple[tuple[str, float], ...],
    filter_universe_id: str,
    training_subject_ids: tuple[str, ...],
    candidate_sender_manifest: tuple[tuple[str, str, tuple[str, ...]], ...],
    parameters: ContrastCommonSenderParameters,
) -> tuple[InteractionLigandContrastSupport, ...]:
    """Fit and jointly Holm-adjust receiver interaction support."""

    contrast_weights = _canonical_contrast_weights(contrast_weights)
    contexts = tuple(context_id for context_id, _ in contrast_weights)
    weight_by_context = dict(contrast_weights)
    minimum_complete = max(2, parameters.min_subjects)
    sender_context = (
        table.groupby(
            [
                "receiver",
                "interaction_id",
                "sender",
                "subject_id",
                "context_id",
            ],
            observed=True,
            sort=True,
        )["ligand_availability"]
        .mean()
        .reset_index()
    )
    interaction_context = (
        sender_context.groupby(
            ["receiver", "interaction_id", "subject_id", "context_id"],
            observed=True,
            sort=True,
        )["ligand_availability"]
        .max(min_count=1)
        .reset_index()
    )
    by_interaction: dict[
        tuple[str, str], dict[tuple[str, str], float]
    ] = {}
    for row in interaction_context.itertuples(index=False):
        if pd.isna(row.ligand_availability):
            continue
        key = (str(row.receiver), str(row.interaction_id))
        by_interaction.setdefault(key, {})[
            (str(row.subject_id), str(row.context_id))
        ] = float(cast(Any, row.ligand_availability))
    raw_supports: list[_RawInteractionContrast] = []
    interaction_keys = tuple(
        (receiver, interaction_id)
        for receiver, interaction_id, _ in candidate_sender_manifest
    )
    for receiver, interaction_id in interaction_keys:
        by_subject_context = by_interaction.get((receiver, interaction_id), {})
        if parameters.contrast_unit == "independent_subject":
            # Independent groups retain every subject/context observation and
            # use a Welch contrast; no artificial pairing or imputation occurs.
            values_by_context = {
                context_id: [
                    (subject_id, by_subject_context[(subject_id, context_id)])
                    for subject_id in training_subject_ids
                    if (subject_id, context_id) in by_subject_context
                ]
                for context_id in contexts
            }
            context_subjects = [
                subject_id
                for context_id in contexts
                for subject_id, _ in values_by_context[context_id]
            ]
            if len(set(context_subjects)) != len(context_subjects):
                raise ContractError(
                    "independent subject contrast requires disjoint subject IDs "
                    "across contexts",
                    code="invalid_common_sender_input",
                    field="subject_id",
                    remediation=(
                        "Use paired_subject for overlapping subjects or provide "
                        "independent subject IDs"
                    ),
                )
            complete_subjects = context_subjects
            effects = [
                float(value)
                for context_id in contexts
                for _, value in values_by_context[context_id]
            ]
            unit_context_ids = [
                context_id
                for context_id in contexts
                for _ in values_by_context[context_id]
            ]
            context_counts = {
                context_id: len(values_by_context[context_id])
                for context_id in contexts
            }
            support_ready = all(
                count >= minimum_complete for count in context_counts.values()
            )
            if not support_ready:
                raw_supports.append(
                    _RawInteractionContrast(
                        receiver=receiver,
                        interaction_id=interaction_id,
                        complete_subject_ids=tuple(complete_subjects),
                        subject_effects=tuple(effects),
                        unit_context_ids=tuple(unit_context_ids),
                        degrees_of_freedom=None,
                        mean_effect=None,
                        sample_standard_error=None,
                        critical_value=None,
                        lower_confidence_bound=None,
                        raw_one_sided_p_value=None,
                    )
                )
            else:
                means = {
                    context_id: math.fsum(
                        value for _, value in values_by_context[context_id]
                    )
                    / context_counts[context_id]
                    for context_id in contexts
                }
                variances = {
                    context_id: math.fsum(
                        (value - means[context_id]) ** 2
                        for _, value in values_by_context[context_id]
                    )
                    / (context_counts[context_id] - 1)
                    for context_id in contexts
                }
                mean_effect = math.fsum(
                    weight_by_context[context_id] * means[context_id]
                    for context_id in contexts
                )
                variance_terms = tuple(
                    weight_by_context[context_id] ** 2
                    * variances[context_id]
                    / context_counts[context_id]
                    for context_id in contexts
                )
                variance = math.fsum(variance_terms)
                sample_standard_error = math.sqrt(max(0.0, variance))
                if sample_standard_error == 0.0:
                    degrees_of_freedom = float(max(1, min(context_counts.values()) - 1))
                else:
                    denominator = math.fsum(
                        term * term / (context_counts[context_id] - 1)
                        for term, context_id in zip(
                            variance_terms, contexts, strict=True
                        )
                    )
                    welch_df = (
                        variance * variance / denominator
                        if denominator > 0.0
                        else float(min(context_counts.values()) - 1)
                    )
                    degrees_of_freedom = max(1.0, welch_df)
                critical_value = float(
                    student_t.ppf(
                        parameters.ligand_contrast_confidence_level,
                        degrees_of_freedom,
                    )
                )
                lower_bound = (
                    mean_effect
                    if sample_standard_error == 0.0
                    else mean_effect - critical_value * sample_standard_error
                )
                raw_p_value = (
                    0.0
                    if sample_standard_error == 0.0
                    and mean_effect > parameters.ligand_contrast_minimum_effect
                    else 1.0
                    if sample_standard_error == 0.0
                    else float(
                        student_t.sf(
                            (
                                mean_effect
                                - parameters.ligand_contrast_minimum_effect
                            )
                            / sample_standard_error,
                            degrees_of_freedom,
                        )
                    )
                )
                raw_supports.append(
                    _RawInteractionContrast(
                        receiver=receiver,
                        interaction_id=interaction_id,
                        complete_subject_ids=tuple(complete_subjects),
                        subject_effects=tuple(effects),
                        unit_context_ids=tuple(unit_context_ids),
                        degrees_of_freedom=degrees_of_freedom,
                        mean_effect=mean_effect,
                        sample_standard_error=sample_standard_error,
                        critical_value=critical_value,
                        lower_confidence_bound=lower_bound,
                        raw_one_sided_p_value=raw_p_value,
                    )
                )
        else:
            complete_subjects = []
            effects = []
            for subject_id in training_subject_ids:
                values = {
                    context_id: by_subject_context.get((subject_id, context_id))
                    for context_id in contexts
                }
                if any(value is None for value in values.values()):
                    continue
                effect = math.fsum(
                    weight_by_context[context_id] * cast(float, values[context_id])
                    for context_id in contexts
                )
                complete_subjects.append(subject_id)
                effects.append(effect)
            n_complete = len(complete_subjects)
            if n_complete < minimum_complete:
                raw_supports.append(
                    _RawInteractionContrast(
                        receiver=receiver,
                        interaction_id=interaction_id,
                        complete_subject_ids=tuple(complete_subjects),
                        subject_effects=tuple(effects),
                        unit_context_ids=(),
                        degrees_of_freedom=None,
                        mean_effect=None,
                        sample_standard_error=None,
                        critical_value=None,
                        lower_confidence_bound=None,
                        raw_one_sided_p_value=None,
                    )
                )
            else:
                mean_effect = math.fsum(effects) / n_complete
                sample_variance = math.fsum(
                    (effect - mean_effect) ** 2 for effect in effects
                ) / (n_complete - 1)
                sample_standard_error = math.sqrt(sample_variance / n_complete)
                degrees_of_freedom = n_complete - 1
                critical_value = float(
                    student_t.ppf(
                        parameters.ligand_contrast_confidence_level,
                        degrees_of_freedom,
                    )
                )
                lower_bound = (
                    mean_effect
                    if sample_variance == 0.0
                    else mean_effect - critical_value * sample_standard_error
                )
                raw_p_value = (
                    0.0
                    if sample_standard_error == 0.0
                    and mean_effect > parameters.ligand_contrast_minimum_effect
                    else 1.0
                    if sample_standard_error == 0.0
                    else float(
                        student_t.sf(
                            (mean_effect - parameters.ligand_contrast_minimum_effect)
                            / sample_standard_error,
                            degrees_of_freedom,
                        )
                    )
                )
                raw_supports.append(
                    _RawInteractionContrast(
                        receiver=receiver,
                        interaction_id=interaction_id,
                        complete_subject_ids=tuple(complete_subjects),
                        subject_effects=tuple(effects),
                        unit_context_ids=(),
                        degrees_of_freedom=degrees_of_freedom,
                        mean_effect=mean_effect,
                        sample_standard_error=sample_standard_error,
                        critical_value=critical_value,
                        lower_confidence_bound=lower_bound,
                        raw_one_sided_p_value=raw_p_value,
                    )
                )

    supports: list[InteractionLigandContrastSupport] = []
    familywise_alpha = _sender_contrast_familywise_alpha(
        parameters.ligand_contrast_confidence_level
    )
    for receiver in sorted({raw.receiver for raw in raw_supports}):
        family = tuple(
            sorted(
                (raw for raw in raw_supports if raw.receiver == receiver),
                key=lambda raw: raw.interaction_id,
            )
        )
        interaction_ids = tuple(raw.interaction_id for raw in family)
        receiver_candidate_manifest = tuple(
            (interaction_id, sender_ids)
            for candidate_receiver, interaction_id, sender_ids in (
                candidate_sender_manifest
            )
            if candidate_receiver == receiver
        )
        family_id = _sender_contrast_family_id(
            receiver=receiver,
            interaction_ids=interaction_ids,
            candidate_sender_ids_by_interaction=receiver_candidate_manifest,
            training_subject_ids=training_subject_ids,
            contrast_weights=contrast_weights,
            filter_universe_id=filter_universe_id,
            minimum_complete_subjects=minimum_complete,
            ligand_contrast_confidence_level=(
                parameters.ligand_contrast_confidence_level
            ),
            ligand_contrast_minimum_effect=(parameters.ligand_contrast_minimum_effect),
            aggregation_policy=(
                _SENDER_CONTRAST_SUPPORT_POLICY_INDEPENDENT
                if parameters.contrast_unit == "independent_subject"
                else _SENDER_CONTRAST_SUPPORT_POLICY
            ),
        )
        adjustments = _holm_step_down_adjustments(
            {raw.interaction_id: raw.raw_one_sided_p_value for raw in family}
        )
        for raw in family:
            holm_rank, adjusted_p = adjustments[raw.interaction_id]
            status = (
                SenderContrastSupportStatus.NOT_ESTIMABLE
                if raw.raw_one_sided_p_value is None
                else SenderContrastSupportStatus.SUPPORTED
                if adjusted_p < familywise_alpha
                else SenderContrastSupportStatus.UNSUPPORTED
            )
            reason_code = (
                (
                    "insufficient_independent_subject_ligand_contrasts"
                    if parameters.contrast_unit == "independent_subject"
                    else "insufficient_complete_subject_ligand_contrasts"
                )
                if status is SenderContrastSupportStatus.NOT_ESTIMABLE
                else None
                if status is SenderContrastSupportStatus.SUPPORTED
                else ("ligand_contrast_holm_adjusted_p_not_below_familywise_alpha")
            )
            supports.append(
                InteractionLigandContrastSupport._from_training(
                    _producer_token=_SENDER_CONTRAST_SUPPORT_PRODUCER_TOKEN,
                    receiver=raw.receiver,
                    interaction_id=raw.interaction_id,
                    complete_subject_ids=raw.complete_subject_ids,
                    subject_effects=raw.subject_effects,
                    minimum_complete_subjects=minimum_complete,
                    contrast_weights=contrast_weights,
                    ligand_contrast_confidence_level=(
                        parameters.ligand_contrast_confidence_level
                    ),
                    ligand_contrast_minimum_effect=(
                        parameters.ligand_contrast_minimum_effect
                    ),
                    degrees_of_freedom=raw.degrees_of_freedom,
                    mean_effect=raw.mean_effect,
                    sample_standard_error=raw.sample_standard_error,
                    critical_value=raw.critical_value,
                    lower_confidence_bound=raw.lower_confidence_bound,
                    raw_one_sided_p_value=raw.raw_one_sided_p_value,
                    multiplicity_method=(_SENDER_CONTRAST_MULTIPLICITY_METHOD),
                    multiplicity_scope=(_SENDER_CONTRAST_MULTIPLICITY_SCOPE),
                    multiplicity_family_id=family_id,
                    multiplicity_family_size=len(family),
                    holm_rank=holm_rank,
                    holm_adjusted_p_value=adjusted_p,
                    status=status,
                    reason_code=reason_code,
                    aggregation_policy=(
                        _SENDER_CONTRAST_SUPPORT_POLICY_INDEPENDENT
                        if parameters.contrast_unit == "independent_subject"
                        else _SENDER_CONTRAST_SUPPORT_POLICY
                    ),
                    unit_context_ids=raw.unit_context_ids,
                )
            )
    return tuple(supports)


@validation_scope()
def fit_contrast_common_sender_functional(
    training_availability: pd.DataFrame,
    *,
    contrast: ContrastSpec,
    context_keys: Sequence[str],
    frozen_interaction_universe: FrozenInteractionUniverse,
    frozen_candidate_sender_manifest: Sequence[Sequence[object]],
    training_input_digest: str,
    parameters: ContrastCommonSenderParameters | None = None,
) -> ContrastCommonSenderFunctional:
    """Fit one sender universe and prevalence prior across training contexts."""

    resolved = parameters or ContrastCommonSenderParameters()
    if not isinstance(resolved, ContrastCommonSenderParameters):
        raise TypeError("parameters must be ContrastCommonSenderParameters or None")
    resolved._require_intact()
    if not isinstance(frozen_interaction_universe, FrozenInteractionUniverse):
        raise TypeError(
            "frozen_interaction_universe must be a FrozenInteractionUniverse"
        )
    frozen_interaction_universe._require_intact()
    filter_universe_id = frozen_interaction_universe.filter_universe_id
    if not isinstance(training_input_digest, str) or not training_input_digest.strip():
        raise ValueError("training_input_digest must be a non-empty string")
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
    frozen_interactions = frozen_interaction_universe.interaction_ids
    candidate_sender_manifest = _validated_candidate_sender_manifest(
        frozen_candidate_sender_manifest,
        frozen_interaction_ids=frozen_interactions,
    )
    observed_candidate_keys = set(
        table[["receiver", "interaction_id", "sender"]]
        .itertuples(index=False, name=None)
    )
    frozen_candidate_keys = {
        (receiver, interaction_id, sender)
        for receiver, interaction_id, sender_ids in candidate_sender_manifest
        for sender in sender_ids
    }
    unknown_candidates = observed_candidate_keys.difference(frozen_candidate_keys)
    if unknown_candidates:
        raise ContractError(
            "training availability contains rows outside the frozen sender manifest",
            code="invalid_common_sender_input",
            field="frozen_candidate_sender_manifest",
            remediation="Fit from the exact pre-frozen candidate universe",
        )
    node_to_context_id: dict[Hashable, str] = {}
    context_columns = tuple(dict.fromkeys((*keys, "context_id")))
    context_rows = table.loc[:, list(context_columns)].drop_duplicates()
    for _, row in context_rows.iterrows():
        if len(keys) == 1:
            node: Hashable = plain_context_value(row[keys[0]])
        else:
            node = canonical_context(row, keys)
        context_identifier = str(row["context_id"])
        previous = node_to_context_id.setdefault(node, context_identifier)
        if previous != context_identifier:
            raise ValueError("context node maps to multiple context IDs")
    missing_nodes = set(contrast.weights).difference(node_to_context_id)
    if missing_nodes:
        raise ValueError("contrast contexts are absent from training availability")
    contrast_context_ids = tuple(
        sorted(
            (
                (node, node_to_context_id[node])
                for node in contrast.weights
            ),
            key=lambda item: canonical_json(item[0]),
        )
    )
    if len({context_id for _, context_id in contrast_context_ids}) != len(
        contrast_context_ids
    ):
        raise ValueError("contrast nodes must map to unique context IDs")
    contrast_weights = _canonical_contrast_weights(
        tuple(
            (context_id, float(contrast.weights[node]))
            for node, context_id in contrast_context_ids
        )
    )
    contexts = tuple(context_id for context_id, _ in contrast_weights)
    table = (
        table.loc[table["context_id"].isin(contexts)]
        .sort_values(list(_INPUT_KEY), kind="stable")
        .reset_index(drop=True)
    )
    if resolved.contrast_unit == "independent_subject":
        # Validate the design on the complete training table before any
        # interaction- or availability-specific NA filtering.  Checking only
        # observed values would allow a mixed design to evade the independent
        # subject contract when an overlapping subject is missing one ligand.
        subjects_by_context = {
            context_id: set(
                table.loc[
                    table["context_id"].eq(context_id), "subject_id"
                ].astype(str)
            )
            for context_id in contexts
        }
        overlap: set[str] = set()
        context_items = tuple(subjects_by_context.items())
        for index, (_, left_subjects) in enumerate(context_items):
            for _, right_subjects in context_items[index + 1 :]:
                overlap.update(left_subjects.intersection(right_subjects))
        if overlap:
            raise ContractError(
                "independent subject contrast requires disjoint subject IDs "
                "across contexts",
                code="invalid_common_sender_input",
                field="subject_id",
                remediation=(
                    "Use paired_subject for overlapping subjects or provide "
                    "independent subject IDs"
                ),
            )
    subjects = tuple(sorted(table["subject_id"].unique()))
    denominator = len(subjects)
    subject_means = table.groupby(
        ["receiver", "interaction_id", "sender", "subject_id"],
        observed=True,
        sort=True,
    )["ligand_availability"].mean()
    candidate_levels = ["receiver", "interaction_id", "sender"]
    observed_subjects = subject_means.notna().groupby(
        level=candidate_levels,
        observed=True,
        sort=True,
    ).sum()
    above_threshold = subject_means.gt(resolved.prevalence_threshold).groupby(
        level=candidate_levels,
        observed=True,
        sort=True,
    ).sum()
    prevalence_stats = {
        cast(tuple[str, str, str], key): (int(n_subjects), int(n_above))
        for key, n_subjects, n_above in zip(
            observed_subjects.index,
            observed_subjects.to_numpy(),
            above_threshold.to_numpy(),
            strict=True,
        )
    }
    priors: list[SenderPrevalencePrior] = []
    for receiver, interaction_id, sender_ids in candidate_sender_manifest:
        for sender in sender_ids:
            n_subjects, n_above = prevalence_stats.get(
                (receiver, interaction_id, sender), (0, 0)
            )
            if n_subjects == 0:
                prior = None
                status = SenderPrevalenceStatus.MISSING_EVIDENCE
                reason = "training_ligand_evidence_missing"
            elif n_subjects < resolved.min_subjects:
                prior = None
                status = SenderPrevalenceStatus.LOW_SUPPORT
                reason = "insufficient_training_subject_support"
            else:
                prior = float(n_above / denominator)
                status = SenderPrevalenceStatus.SUPPORTED
                reason = None
            priors.append(
                SenderPrevalencePrior(
                    receiver=receiver,
                    interaction_id=interaction_id,
                    sender=sender,
                    prevalence_prior=prior,
                    n_subjects=n_subjects,
                    status=status,
                    reason_code=reason,
                )
            )
    contrast_supports = _fit_interaction_ligand_contrast_supports(
        table,
        contrast_weights=contrast_weights,
        filter_universe_id=filter_universe_id,
        training_subject_ids=subjects,
        candidate_sender_manifest=candidate_sender_manifest,
        parameters=resolved,
    )
    training_digest = _training_availability_digest(
        table,
        contrast_context_ids=contrast_context_ids,
        contrast_weights=contrast_weights,
        filter_universe_id=filter_universe_id,
        frozen_interaction_ids=frozen_interactions,
        candidate_sender_manifest=candidate_sender_manifest,
    )
    return ContrastCommonSenderFunctional(
        _producer_token=_SENDER_FUNCTIONAL_PRODUCER_TOKEN,
        contrast=contrast,
        contrast_context_ids=contrast_context_ids,
        training_subject_ids=subjects,
        filter_universe_id=filter_universe_id,
        frozen_interaction_ids=frozen_interactions,
        candidate_sender_manifest=candidate_sender_manifest,
        training_availability_digest=training_digest,
        training_input_digest=training_input_digest.strip(),
        parameters=resolved,
        candidate_priors=tuple(priors),
        contrast_supports=contrast_supports,
    )


@validation_scope()
def interaction_ligand_contrast_gates(
    functional: ContrastCommonSenderFunctional,
    interactions: Sequence[tuple[str, str]],
) -> tuple[InteractionLigandContrastGate, ...]:
    """Return frozen interaction gates after one functional integrity check."""

    if not isinstance(functional, ContrastCommonSenderFunctional):
        raise TypeError("functional must be ContrastCommonSenderFunctional")
    functional._require_intact()
    queries: list[tuple[str, str]] = []
    for receiver, interaction_id in interactions:
        if not isinstance(receiver, str) or not receiver.strip():
            raise ValueError("receiver must be a non-empty identifier")
        if not isinstance(interaction_id, str) or not interaction_id.strip():
            raise ValueError("interaction_id must be a non-empty identifier")
        queries.append((receiver.strip(), interaction_id.strip()))
    support_by_key = {
        (support.receiver, support.interaction_id): support
        for support in functional.contrast_supports
    }
    gates: list[InteractionLigandContrastGate] = []
    for normalized_receiver, normalized_interaction in queries:
        support = support_by_key.get((normalized_receiver, normalized_interaction))
        if support is None:
            gates.append(
                InteractionLigandContrastGate(
                    sender_functional_id=functional.sender_functional_id,
                    receiver=normalized_receiver,
                    interaction_id=normalized_interaction,
                    gate=None,
                    status=SenderContrastSupportStatus.NOT_ESTIMABLE,
                    reason_code="interaction_ligand_contrast_support_absent",
                    support_ids=(),
                )
            )
            continue
        support._require_intact()
        status = SenderContrastSupportStatus(support.status)
        gates.append(
            InteractionLigandContrastGate(
                sender_functional_id=functional.sender_functional_id,
                receiver=normalized_receiver,
                interaction_id=normalized_interaction,
                gate=(
                    1.0
                    if status is SenderContrastSupportStatus.SUPPORTED
                    else 0.0
                    if status is SenderContrastSupportStatus.UNSUPPORTED
                    else None
                ),
                status=status,
                reason_code=support.reason_code,
                support_ids=(support.support_id,),
            )
        )
    return tuple(gates)


def interaction_ligand_contrast_gate(
    functional: ContrastCommonSenderFunctional,
    receiver: str,
    interaction_id: str,
) -> InteractionLigandContrastGate:
    """Return one frozen interaction gate without reading held-out values."""

    return interaction_ligand_contrast_gates(
        functional,
        ((receiver, interaction_id),),
    )[0]


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


@validation_scope()
def apply_contrast_common_sender_functional(
    functional: ContrastCommonSenderFunctional,
    sample_availability: pd.DataFrame,
) -> CommonSenderApplication:
    """Apply frozen priors to sample-local ligand evidence without refitting."""

    if not isinstance(functional, ContrastCommonSenderFunctional):
        raise TypeError("functional must be ContrastCommonSenderFunctional")
    functional._require_intact()
    table = _validated_availability(sample_availability)
    table = table.loc[table["context_id"].isin(functional.context_ids)].copy()
    if table.empty:
        return CommonSenderApplication(
            pd.DataFrame(columns=COMMON_SENDER_APPLICATION_COLUMNS), functional
        )
    mutable_priors_by_group: dict[
        tuple[str, str], list[SenderPrevalencePrior]
    ] = {}
    for prior in functional.candidate_priors:
        mutable_priors_by_group.setdefault(
            (prior.receiver, prior.interaction_id), []
        ).append(prior)
    priors_by_group = {
        key: tuple(priors) for key, priors in mutable_priors_by_group.items()
    }
    output: list[dict[str, Any]] = []
    group_columns = [*COMMON_SENDER_GROUP_COLUMNS]
    for group_key, group in table.groupby(group_columns, observed=True, sort=True):
        sample_id, subject_id, context_id, receiver, interaction_id = group_key
        priors = priors_by_group.get((str(receiver), str(interaction_id)))
        if priors is None:
            continue
        candidate_senders = {prior.sender for prior in priors}
        local = {
            str(row.sender): (
                None
                if pd.isna(row.ligand_availability)
                else float(cast(Any, row.ligand_availability))
            )
            for row in group.itertuples(index=False)
            if str(row.sender) in candidate_senders
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
    "interaction_ligand_contrast_gate",
    "interaction_ligand_contrast_gates",
]
