"""Independent forward/reverse LR views over one directional cross-fit.

This module is deliberately an adapter, not another fitting stage.  Each
channel is copied from the exact ``FamilyCommonScoringApplication`` produced
for its registered contrast.  The forward and reverse values are therefore
valid independent activation-compatible LR views, but are not a signed,
common-scale, or comparable pair.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.attribution import (
    DirectionalContrastPairSpec,
    FrozenReceiverFamilyLRHypothesisUniverse,
)
from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import node_context_fields

from .crossfit import CrossFitArtifacts, CrossFitFoldArtifacts
from .directional import DirectionalCrossFitBinding

DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION = "directional_integrated_lr_v3"
DIRECTIONAL_INTEGRATED_LR_SEMANTICS = (
    "independent_directional_activation_compatible_molecular_lr_full_grid_v3"
)
DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK = (
    "directional_integrated_lr_full_frozen_grid_descriptive"
)

DIRECTIONAL_INTEGRATED_LR_COLUMNS = (
    "crossfit_id",
    "crossfit_spec_id",
    "pair_spec_id",
    "directional_lr_hypothesis_universe_id",
    "fold_id",
    "receiver",
    "receiver_training_support_id",
    "receiver_training_support_status",
    "receiver_training_support_reason_code",
    "directional_binding_id",
    "channel_role",
    "channel",
    "contrast_manifest_id",
    "contrast_name",
    "design_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "family_id",
    "driver_id",
    "interaction_id",
    "molecular_lr_equivalence_id",
    "mode",
    "receiver_family_lr_membership_id",
    "receiver_family_lr_hypothesis_id",
    "receiver_family_lr_opportunity_id",
    "integrated_lr_score",
    "family_core_strength",
    "within_family_lr_weight",
    "source_status",
    "source_reason_code",
    "status",
    "reason_code",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "family_common_member_scores_digest",
    "family_common_family_scores_digest",
    "edge_evidence_digest",
    "incremental_training_artifact_id",
    "incremental_application_id",
    "score_version",
    "semantics",
    "analysis_track",
    "experimental",
    "source_agnostic",
    "cross_channel_comparable",
    "paired_score_comparison_allowed",
    "active_inhibition_allowed",
    "supports_active_inhibition_claim",
    "formal_inference_allowed",
)

DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS = (
    "crossfit_id",
    "crossfit_spec_id",
    "pair_spec_id",
    "directional_lr_hypothesis_universe_id",
    "receiver_family_lr_opportunity_axis_id",
    "receiver_universe_id",
    "receiver_axis_id",
    "fold_id",
    "receiver",
    "receiver_training_support_id",
    "receiver_training_support_status",
    "receiver_training_support_reason_code",
    "directional_binding_id",
    "forward_response_id",
    "reverse_response_id",
    "forward_incremental_training_artifact_id",
    "reverse_incremental_training_artifact_id",
    "forward_incremental_application_id",
    "reverse_incremental_application_id",
    "forward_family_common_functional_id",
    "reverse_family_common_functional_id",
    "forward_family_common_application_id",
    "reverse_family_common_application_id",
    "forward_family_common_binding_id",
    "reverse_family_common_binding_id",
    "forward_channel",
    "reverse_channel",
    "status",
    "reason_code",
    "lr_hypothesis_grid_status",
    "lr_hypothesis_grid_reason_code",
    "n_forward_score_rows",
    "n_reverse_score_rows",
    "paired_score_comparison_allowed",
    "active_inhibition_allowed",
    "supports_active_inhibition_claim",
    "formal_inference_allowed",
)

_OPPORTUNITY_BINDING_PARENT_COLUMNS = (
    "forward_response_id",
    "reverse_response_id",
    "forward_incremental_training_artifact_id",
    "reverse_incremental_training_artifact_id",
    "forward_incremental_application_id",
    "reverse_incremental_application_id",
)
_OPPORTUNITY_FAMILY_PARENT_COLUMNS = (
    "forward_family_common_functional_id",
    "reverse_family_common_functional_id",
    "forward_family_common_application_id",
    "reverse_family_common_application_id",
    "forward_family_common_binding_id",
    "reverse_family_common_binding_id",
)
_OPPORTUNITY_PARENT_COLUMNS = (
    *_OPPORTUNITY_BINDING_PARENT_COLUMNS,
    *_OPPORTUNITY_FAMILY_PARENT_COLUMNS,
)

_FITTED_SCORE_PARENT_COLUMNS = (
    "directional_binding_id",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "family_common_member_scores_digest",
    "family_common_family_scores_digest",
    "edge_evidence_digest",
    "incremental_training_artifact_id",
    "incremental_application_id",
)

_ROW_KEY = (
    "fold_id",
    "receiver",
    "channel_role",
    "sample_id",
    "subject_id",
    "context_id",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
)
_CHANNEL_KEY = tuple(column for column in _ROW_KEY if column != "channel_role")
_SOURCE_KEY = tuple(column for column in _CHANNEL_KEY if column != "fold_id")
_STATUS_VALUES = frozenset({"observed", "structural_zero", "not_estimable"})
_SOURCE_STATUS_VALUES = frozenset({"ok", "structural_zero", "not_estimable"})
_SCHEMA_VERSION = "4.0.0"
_IDENTITY_SCHEMA_VERSION = "4"
_PRODUCER = "crychic.workflow.directional_integrated_lr.v4"
_COLLECTION_MARKER = "crychic.directional_integrated_lr_collection.v4"
_PAIR_NE_REASON = "directional_pair_not_estimable"
_ABSENT_RECEIVER_REASON = "receiver_absent_in_outer_training"
_FULL_GRID_SCOPE = "full_run_root_frozen_receiver_family_lr_hypothesis_universe"


def _source_mismatch(message: str, *, field_name: str) -> ContractError:
    return ContractError(
        message,
        code="directional_integrated_lr_source_mismatch",
        field=field_name,
        remediation=(
            "Use one intact CrossFitArtifacts object with its exact directional "
            "binding and family-common parents"
        ),
    )


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    result = str(value)
    return result if result else None


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    numeric = float(cast(Any, value))
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite or missing")
    return numeric


def _cell_token(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing", "value": None}
        if not math.isfinite(value):
            raise ValueError("directional LR tables cannot contain infinity")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": canonical_json(value),
    }


def _table_digest(table: pd.DataFrame) -> str:
    rows = [
        [_cell_token(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    rows.sort(key=canonical_json)
    return str(
        stable_id(
            "directional_integrated_lr_table",
            {"columns": list(table.columns), "rows": rows},
            schema_version=_IDENTITY_SCHEMA_VERSION,
            digest_length=64,
        )
    )


def _status_counts(table: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    return tuple(
        (status, int(table["status"].eq(status).sum()))
        for status in ("observed", "structural_zero", "not_estimable")
    )


def _opportunity_status_counts(
    table: pd.DataFrame,
) -> tuple[tuple[str, int], ...]:
    counts = table["status"].astype(str).value_counts().to_dict()
    return tuple(
        (str(status), int(count))
        for status, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _score_row_counts(table: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for fold_id, receiver, role in table.loc[
        :, ["fold_id", "receiver", "channel_role"]
    ].itertuples(index=False, name=None):
        key = (str(fold_id), str(receiver), str(role))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _score_parent_id(
    table: pd.DataFrame,
    *,
    fold_id: str,
    receiver: str,
    role: str,
    column: str,
) -> str:
    selected = table.loc[
        table["fold_id"].eq(fold_id)
        & table["receiver"].eq(receiver)
        & table["channel_role"].eq(role),
        column,
    ]
    values = {_optional_text(value) for value in selected}
    if len(values) != 1 or None in values:
        raise _source_mismatch(
            "Directional LR score rows do not retain one channel parent",
            field_name=column,
        )
    return cast(str, next(iter(values)))


def _opportunity_parent_ids(
    table: pd.DataFrame,
    *,
    fold_id: str,
    receiver: str,
    binding: DirectionalCrossFitBinding,
) -> dict[str, object]:
    return {
        "forward_response_id": binding.forward_response_id,
        "reverse_response_id": binding.reverse_response_id,
        "forward_incremental_training_artifact_id": (
            binding.forward_training_artifact_id
        ),
        "reverse_incremental_training_artifact_id": (
            binding.reverse_training_artifact_id
        ),
        "forward_incremental_application_id": binding.forward_application_id,
        "reverse_incremental_application_id": binding.reverse_application_id,
        **{
            f"{role}_{suffix}": _score_parent_id(
                table,
                fold_id=fold_id,
                receiver=receiver,
                role=role,
                column=column,
            )
            for role in ("forward", "reverse")
            for suffix, column in (
                ("family_common_functional_id", "family_common_functional_id"),
                ("family_common_application_id", "family_common_application_id"),
                ("family_common_binding_id", "family_common_binding_id"),
            )
        },
    }


def _pair_for_id(
    artifacts: CrossFitArtifacts, pair_spec_id: str
) -> DirectionalContrastPairSpec:
    _required_name(pair_spec_id, field_name="pair_spec_id")
    matches = tuple(
        pair
        for pair in artifacts.spec.directional_pairs
        if pair.pair_spec_id == pair_spec_id
    )
    if len(matches) != 1:
        raise _source_mismatch(
            "Cross-fit does not contain exactly one requested directional pair",
            field_name="pair_spec_id",
        )
    pair = matches[0]
    pair._require_intact()
    return pair


def _fold_map(artifacts: CrossFitArtifacts) -> dict[str, CrossFitFoldArtifacts]:
    result = {fold.fold_id: fold for fold in artifacts.folds}
    if len(result) != len(artifacts.folds) or not result:
        raise _source_mismatch(
            "Cross-fit fold identities are invalid", field_name="folds"
        )
    return result


def _channel_specs(
    pair: DirectionalContrastPairSpec,
) -> tuple[tuple[str, str, Any], ...]:
    return (
        ("forward", pair.forward_channel_name, pair.forward_contrast),
        ("reverse", pair.reverse_channel_name, pair.reverse_contrast),
    )


def _lr_hypothesis_universe(
    artifacts: CrossFitArtifacts,
) -> FrozenReceiverFamilyLRHypothesisUniverse:
    universe = artifacts.directional_lr_hypothesis_universe
    if not isinstance(universe, FrozenReceiverFamilyLRHypothesisUniverse):
        raise _source_mismatch(
            "Directional cross-fit lacks its frozen LR hypothesis universe",
            field_name="directional_lr_hypothesis_universe",
        )
    universe._require_intact()
    if (
        universe.receiver_ids != artifacts.receiver_universe.receiver_ids
        or universe.receiver_universe_id != artifacts.receiver_universe.universe_id
        or universe.receiver_axis_id != artifacts.receiver_universe.receiver_axis_id
        or universe.root_input_identity_id != artifacts.root_input_identity.identity_id
        or universe.root_input_digest != artifacts.root_input_digest
    ):
        raise _source_mismatch(
            "Frozen directional LR universe differs from the cross-fit run root",
            field_name="directional_lr_hypothesis_universe",
        )
    return universe


def _design_application(
    fold: CrossFitFoldArtifacts,
    *,
    contrast: Any,
    role: str,
) -> tuple[Any, tuple[tuple[str, str, str], ...]]:
    matches = tuple(
        (encoder, application)
        for encoder, application in zip(
            fold.design_encoders,
            fold.design_applications,
            strict=True,
        )
        if encoder.contrast.name == contrast.name
    )
    if len(matches) != 1:
        raise _source_mismatch(
            "Directional channel lacks one exact held-out design application",
            field_name=f"{role}_design_application",
        )
    encoder, application = matches[0]
    encoder.to_dict()
    application.require_compatible(encoder)
    if (
        encoder.contrast.to_dict() != contrast.to_dict()
        or application.application_scope != "heldout"
    ):
        raise _source_mismatch(
            "Directional held-out design lineage differs from its contrast",
            field_name=f"{role}_design_application",
        )
    context_ids = {
        node_context_fields(node, encoder.context_keys)[0]
        for node in encoder.contrast.weights
    }
    rows = tuple(
        (str(sample_id), str(subject_id), str(context_id))
        for sample_id, subject_id, context_id in zip(
            application.sample_ids,
            application.sample_subject_ids,
            application.sample_context_ids,
            strict=True,
        )
        if str(context_id) in context_ids
    )
    if (
        not rows
        or len(rows) != len({sample_id for sample_id, _, _ in rows})
        or {context_id for _, _, context_id in rows} != context_ids
    ):
        raise _source_mismatch(
            "Directional held-out design lacks its exact contrast context domain",
            field_name=f"{role}_design_application",
        )
    return application, rows


def _frozen_lr_opportunities(
    universe: FrozenReceiverFamilyLRHypothesisUniverse,
    *,
    receiver: str,
) -> tuple[dict[str, str], ...]:
    molecular_by_membership = {
        membership.membership_id: membership.molecular_lr_equivalence_id
        for membership in universe.memberships
    }
    rows = tuple(
        {
            "receiver": candidate_receiver,
            "interaction_id": interaction_id,
            "driver_id": driver_id,
            "family_id": family_id,
            "molecular_lr_equivalence_id": molecular_by_membership[membership_id],
            "receiver_family_lr_membership_id": membership_id,
            "mode": mode,
            "receiver_family_lr_hypothesis_id": hypothesis_id,
            "receiver_family_lr_opportunity_id": opportunity_id,
        }
        for (
            candidate_receiver,
            interaction_id,
            driver_id,
            family_id,
            membership_id,
            mode,
            hypothesis_id,
            opportunity_id,
        ) in universe.opportunity_ids
        if candidate_receiver == receiver
    )
    if not rows:
        raise _source_mismatch(
            "Receiver is absent from the frozen directional LR universe",
            field_name="directional_lr_hypothesis_universe.opportunity_ids",
        )
    return rows


def _key_tuple(
    record: Mapping[Hashable, object],
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(str(record[column]) for column in columns)


def _unique_by_key(
    values: tuple[Any, ...], key_name: str
) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for value in values:
        receiver = getattr(value, "receiver", None)
        contrast = getattr(value, "contrast_name", None)
        key = (str(contrast), str(receiver))
        if key in result:
            raise _source_mismatch(
                f"{key_name} contains duplicate contrast/receiver parents",
                field_name=key_name,
            )
        result[key] = value
    return result


def _directional_bindings(
    fold: CrossFitFoldArtifacts,
    pair: DirectionalContrastPairSpec,
) -> dict[str, DirectionalCrossFitBinding]:
    result: dict[str, DirectionalCrossFitBinding] = {}
    for binding in fold.directional_response_bindings:
        if binding.pair_spec_id != pair.pair_spec_id:
            continue
        binding._require_intact()
        receiver = str(binding.receiver)
        if receiver in result:
            raise _source_mismatch(
                "Directional bindings duplicate a pair/receiver key",
                field_name="directional_response_bindings",
            )
        result[receiver] = binding
    expected = {
        str(record.receiver_id)
        for record in fold.receiver_training_support
        if record.status == "observed"
    }
    if set(result) != expected:
        raise _source_mismatch(
            "Directional bindings do not cover every outer-training-supported receiver",
            field_name="directional_response_bindings",
        )
    return result


def _parent_chain(
    fold: CrossFitFoldArtifacts,
    *,
    contrast_name: str,
    contrast: Any,
    receiver: str,
    directional: DirectionalCrossFitBinding,
    role: str,
) -> tuple[Any, Any, Any, Any, Any]:
    common_parents: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    for functional, application, common_binding in zip(
        fold.family_common_functionals,
        fold.family_common_applications,
        fold.family_common_bindings,
        strict=True,
    ):
        key = (str(functional.contrast_name), str(functional.receiver))
        if key in common_parents:
            raise _source_mismatch(
                "Family-common parents duplicate a contrast/receiver key",
                field_name="family_common_functionals",
            )
        common_parents[key] = (functional, application, common_binding)
    key = (contrast_name, receiver)
    try:
        functional, application, common_binding = common_parents[key]
    except KeyError as error:
        raise _source_mismatch(
            "Directional channel lacks its family-common parent chain",
            field_name=f"{role}_family_common_parent",
        ) from error
    functional._require_intact()
    application._require_intact()
    common_binding._require_intact()
    if application.functional.family_common_functional_id != (
        functional.family_common_functional_id
    ) or common_binding.family_common_functional_id != (
        functional.family_common_functional_id
    ):
        raise _source_mismatch(
            "Family-common functional/application/binding lineage differs",
            field_name=f"{role}_family_common_parent",
        )
    models = _unique_by_key(
        tuple(fold.receiver_incremental_models), "receiver_incremental_models"
    )
    incremental_apps: dict[tuple[str, str], Any] = {}
    for model, incremental_app in zip(
        fold.receiver_incremental_models,
        fold.receiver_incremental_applications,
        strict=True,
    ):
        incremental_key = (str(model.contrast_name), str(model.receiver))
        if incremental_key in incremental_apps:
            raise _source_mismatch(
                "Receiver incremental applications duplicate a contrast/receiver key",
                field_name="receiver_incremental_applications",
            )
        incremental_apps[incremental_key] = incremental_app
    try:
        model = models[key]
        incremental_app = incremental_apps[key]
    except KeyError as error:
        raise _source_mismatch(
            "Directional channel lacks its incremental parent chain",
            field_name=f"{role}_incremental_parent",
        ) from error
    model._require_intact()
    incremental_app._require_intact()
    expected_training_id = (
        directional.forward_training_artifact_id
        if role == "forward"
        else directional.reverse_training_artifact_id
    )
    expected_application_id = (
        directional.forward_application_id
        if role == "forward"
        else directional.reverse_application_id
    )
    if (
        functional.fold_id != fold.fold_id
        or functional.receiver != receiver
        or functional.contrast_name != contrast_name
        or functional.contrast_manifest_id
        != stable_id("contrast_manifest", contrast.to_dict())
        or functional.receiver_incremental_training_artifact_id
        != model.training_artifact_id
        or model.training_artifact_id != expected_training_id
        or incremental_app.application_id != expected_application_id
        or incremental_app.training_artifact_id != model.training_artifact_id
    ):
        raise _source_mismatch(
            "Directional and family-common incremental parent lineage differs",
            field_name=f"{role}_incremental_parent",
        )
    return functional, application, common_binding, model, incremental_app


def _validate_source_grid(
    table: pd.DataFrame,
    *,
    role: str,
    receiver: str,
    design_application: Any,
    design_rows: tuple[tuple[str, str, str], ...],
    universe: FrozenReceiverFamilyLRHypothesisUniverse,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "sender_unresolved_strength",
        "family_core_strength",
        "within_family_lr_weight",
        "status",
        "reason_code",
    }
    if not required.issubset(table.columns):
        raise _source_mismatch(
            "Family-common member table lacks directional LR columns",
            field_name=f"{role}_member_scores",
        )
    source = table.copy(deep=True)
    if source.duplicated(list(_SOURCE_KEY)).any():
        raise _source_mismatch(
            "Family-common member table contains duplicate LR rows",
            field_name=f"{role}_member_scores",
        )
    design_application._require_intact()
    opportunities = _frozen_lr_opportunities(universe, receiver=receiver)
    expected_keys = {
        (
            receiver,
            str(sample_id),
            str(subject_id),
            str(context_id),
            opportunity["family_id"],
            opportunity["driver_id"],
            opportunity["interaction_id"],
            opportunity["mode"],
        )
        for sample_id, subject_id, context_id in design_rows
        for opportunity in opportunities
    }
    observed_keys = {
        _key_tuple(record, _SOURCE_KEY) for record in source.to_dict(orient="records")
    }
    if observed_keys != expected_keys:
        raise _source_mismatch(
            "Family-common member rows do not exactly cover held-out samples "
            "by the frozen mapped LR-mode grid",
            field_name=f"{role}_member_scores",
        )
    for status in source["status"].astype(str):
        if status not in _SOURCE_STATUS_VALUES:
            raise _source_mismatch(
                "Family-common member table has an unsupported status",
                field_name=f"{role}_member_scores.status",
            )
    mapped_status = source["status"].astype(str).replace({"ok": "observed"})
    score = source["sender_unresolved_strength"].map(
        lambda value: _optional_float(value, field_name="sender_unresolved_strength")
    )
    for value, status, reason in zip(
        score,
        mapped_status,
        source["reason_code"],
        strict=True,
    ):
        reason_text = _optional_text(reason)
        if status == "observed" and (
            value is None or reason_text is not None or not 0.0 <= value <= 1.0
        ):
            raise _source_mismatch(
                "Observed directional member score is missing or out of range",
                field_name=f"{role}_member_scores.sender_unresolved_strength",
            )
        if status == "structural_zero" and (value != 0.0 or reason_text is None):
            raise _source_mismatch(
                "Structural-zero directional member score is not zero",
                field_name=f"{role}_member_scores.sender_unresolved_strength",
            )
        if status == "not_estimable" and (value is not None or reason_text is None):
            raise _source_mismatch(
                "Not-estimable directional member score is not missing",
                field_name=f"{role}_member_scores.sender_unresolved_strength",
            )
    opportunity_by_key = {
        (row["interaction_id"], row["mode"]): row for row in opportunities
    }
    membership_ids: list[str] = []
    hypothesis_ids: list[str] = []
    opportunity_ids: list[str] = []
    molecular_lr_equivalence_ids: list[str] = []
    for record in source.to_dict(orient="records"):
        key = (str(record["interaction_id"]), str(record["mode"]))
        try:
            opportunity = opportunity_by_key[key]
        except KeyError as error:  # pragma: no cover - exact-grid guard
            raise _source_mismatch(
                "Family-common member row is outside the frozen LR universe",
                field_name=f"{role}_member_scores",
            ) from error
        if (
            str(record["receiver"]) != receiver
            or str(record["family_id"]) != opportunity["family_id"]
            or str(record["driver_id"]) != opportunity["driver_id"]
        ):
            raise _source_mismatch(
                "Family-common LR membership differs from the frozen universe",
                field_name=f"{role}_member_scores",
            )
        membership_ids.append(opportunity["receiver_family_lr_membership_id"])
        hypothesis_ids.append(opportunity["receiver_family_lr_hypothesis_id"])
        opportunity_ids.append(opportunity["receiver_family_lr_opportunity_id"])
        molecular_lr_equivalence_ids.append(opportunity["molecular_lr_equivalence_id"])
    source["_normalized_status"] = mapped_status.to_numpy()
    source["_normalized_score"] = score.to_numpy()
    source["_membership_id"] = membership_ids
    source["_hypothesis_id"] = hypothesis_ids
    source["_opportunity_id"] = opportunity_ids
    source["_molecular_lr_equivalence_id"] = molecular_lr_equivalence_ids
    return source


def _lineage_payload(
    *,
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
    binding: DirectionalCrossFitBinding,
    functional: Any,
    application: Any,
    common_binding: Any,
    model: Any,
    incremental_app: Any,
    receiver_support: Any,
    design_application: Any,
    universe: FrozenReceiverFamilyLRHypothesisUniverse,
    role: str,
    channel: str,
    contrast: Any,
) -> dict[str, object]:
    return {
        "crossfit_id": artifacts.crossfit_id,
        "crossfit_spec_id": artifacts.spec.spec_id,
        "pair_spec_id": pair.pair_spec_id,
        "directional_lr_hypothesis_universe_id": universe.universe_id,
        "fold_id": functional.fold_id,
        "receiver": functional.receiver,
        "receiver_training_support_id": receiver_support.support_record_id,
        "receiver_training_support_status": receiver_support.status.value,
        "receiver_training_support_reason_code": receiver_support.reason_code,
        "directional_binding_id": binding.binding_id,
        "channel_role": role,
        "channel": channel,
        "contrast_manifest_id": functional.contrast_manifest_id,
        "contrast_name": contrast.name,
        "design_application_id": design_application.application_id,
        "family_common_functional_id": functional.family_common_functional_id,
        "family_common_application_id": application.application_id,
        "family_common_binding_id": common_binding.binding_id,
        "family_common_member_scores_digest": application.member_scores_digest,
        "family_common_family_scores_digest": application.family_scores_digest,
        "edge_evidence_digest": common_binding.edge_evidence_digest,
        "incremental_training_artifact_id": model.training_artifact_id,
        "incremental_application_id": incremental_app.application_id,
        "score_version": DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION,
        "semantics": DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
        "analysis_track": DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK,
        "experimental": True,
        "source_agnostic": False,
        "cross_channel_comparable": False,
        "paired_score_comparison_allowed": False,
        "active_inhibition_allowed": False,
        "supports_active_inhibition_claim": False,
        "formal_inference_allowed": False,
    }


def _channel_rows(
    *,
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
    binding: DirectionalCrossFitBinding,
    functional: Any,
    application: Any,
    common_binding: Any,
    model: Any,
    incremental_app: Any,
    receiver_support: Any,
    design_application: Any,
    design_rows: tuple[tuple[str, str, str], ...],
    universe: FrozenReceiverFamilyLRHypothesisUniverse,
    role: str,
    channel: str,
    contrast: Any,
) -> pd.DataFrame:
    source = _validate_source_grid(
        application.member_scores,
        role=role,
        receiver=str(functional.receiver),
        design_application=design_application,
        design_rows=design_rows,
        universe=universe,
    )
    rows: list[dict[str, object]] = []
    lineage = _lineage_payload(
        artifacts=artifacts,
        pair=pair,
        binding=binding,
        functional=functional,
        application=application,
        common_binding=common_binding,
        model=model,
        incremental_app=incremental_app,
        receiver_support=receiver_support,
        design_application=design_application,
        universe=universe,
        role=role,
        channel=channel,
        contrast=contrast,
    )
    pair_observed = binding.status == "observed"
    for record in source.to_dict(orient="records"):
        raw_status = str(record["_normalized_status"])
        raw_score = cast(float | None, record["_normalized_score"])
        if pair_observed:
            status = raw_status
            score = raw_score
            reason = _optional_text(record["reason_code"])
        else:
            status = "not_estimable"
            score = None
            reason = binding.reason_code or _PAIR_NE_REASON
        row = {
            **lineage,
            "sample_id": record["sample_id"],
            "subject_id": record["subject_id"],
            "context_id": record["context_id"],
            "family_id": record["family_id"],
            "driver_id": record["driver_id"],
            "interaction_id": record["interaction_id"],
            "molecular_lr_equivalence_id": record["_molecular_lr_equivalence_id"],
            "mode": record["mode"],
            "receiver_family_lr_membership_id": record["_membership_id"],
            "receiver_family_lr_hypothesis_id": record["_hypothesis_id"],
            "receiver_family_lr_opportunity_id": record["_opportunity_id"],
            "integrated_lr_score": score,
            "family_core_strength": record["family_core_strength"],
            "within_family_lr_weight": record["within_family_lr_weight"],
            "source_status": raw_status,
            "source_reason_code": _optional_text(record["reason_code"]),
            "status": status,
            "reason_code": reason,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=DIRECTIONAL_INTEGRATED_LR_COLUMNS)


def _absent_receiver_channel_rows(
    *,
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
    fold: CrossFitFoldArtifacts,
    receiver_support: Any,
    design_application: Any,
    design_rows: tuple[tuple[str, str, str], ...],
    universe: FrozenReceiverFamilyLRHypothesisUniverse,
    role: str,
    channel: str,
    contrast: Any,
) -> pd.DataFrame:
    if (
        receiver_support.status.value != "not_estimable"
        or receiver_support.reason_code != _ABSENT_RECEIVER_REASON
    ):
        raise _source_mismatch(
            "Synthetic directional LR rows require an absent-training receiver",
            field_name="receiver_training_support",
        )
    opportunities = _frozen_lr_opportunities(
        universe,
        receiver=receiver_support.receiver_id,
    )
    rows: list[dict[str, object]] = []
    for sample_id, subject_id, context_id in design_rows:
        for opportunity in opportunities:
            rows.append(
                {
                    "crossfit_id": artifacts.crossfit_id,
                    "crossfit_spec_id": artifacts.spec.spec_id,
                    "pair_spec_id": pair.pair_spec_id,
                    "directional_lr_hypothesis_universe_id": universe.universe_id,
                    "fold_id": fold.fold_id,
                    "receiver": receiver_support.receiver_id,
                    "receiver_training_support_id": (
                        receiver_support.support_record_id
                    ),
                    "receiver_training_support_status": (receiver_support.status.value),
                    "receiver_training_support_reason_code": (
                        receiver_support.reason_code
                    ),
                    "directional_binding_id": None,
                    "channel_role": role,
                    "channel": channel,
                    "contrast_manifest_id": stable_id(
                        "contrast_manifest", contrast.to_dict()
                    ),
                    "contrast_name": contrast.name,
                    "design_application_id": design_application.application_id,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "context_id": context_id,
                    "family_id": opportunity["family_id"],
                    "driver_id": opportunity["driver_id"],
                    "interaction_id": opportunity["interaction_id"],
                    "molecular_lr_equivalence_id": opportunity[
                        "molecular_lr_equivalence_id"
                    ],
                    "mode": opportunity["mode"],
                    "receiver_family_lr_membership_id": opportunity[
                        "receiver_family_lr_membership_id"
                    ],
                    "receiver_family_lr_hypothesis_id": opportunity[
                        "receiver_family_lr_hypothesis_id"
                    ],
                    "receiver_family_lr_opportunity_id": opportunity[
                        "receiver_family_lr_opportunity_id"
                    ],
                    "integrated_lr_score": np.nan,
                    "family_core_strength": np.nan,
                    "within_family_lr_weight": np.nan,
                    "source_status": None,
                    "source_reason_code": None,
                    "status": "not_estimable",
                    "reason_code": _ABSENT_RECEIVER_REASON,
                    "family_common_functional_id": None,
                    "family_common_application_id": None,
                    "family_common_binding_id": None,
                    "family_common_member_scores_digest": None,
                    "family_common_family_scores_digest": None,
                    "edge_evidence_digest": None,
                    "incremental_training_artifact_id": None,
                    "incremental_application_id": None,
                    "score_version": DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION,
                    "semantics": DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
                    "analysis_track": DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK,
                    "experimental": True,
                    "source_agnostic": False,
                    "cross_channel_comparable": False,
                    "paired_score_comparison_allowed": False,
                    "active_inhibition_allowed": False,
                    "supports_active_inhibition_claim": False,
                    "formal_inference_allowed": False,
                }
            )
    if not rows:  # pragma: no cover - producer universe/design invariants
        raise _source_mismatch(
            "Absent receiver LR grid is empty",
            field_name="scores",
        )
    return pd.DataFrame(rows, columns=DIRECTIONAL_INTEGRATED_LR_COLUMNS)


def _materialize_table(
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
) -> tuple[pd.DataFrame, tuple[dict[str, object], ...]]:
    universe = _lr_hypothesis_universe(artifacts)
    parts: list[pd.DataFrame] = []
    parent_rows: list[dict[str, object]] = []
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        directional = _directional_bindings(fold, pair)
        for receiver_support in fold.receiver_training_support:
            receiver_support._require_intact()
            receiver = receiver_support.receiver_id
            pair_binding = directional.get(receiver)
            for role, channel, contrast in _channel_specs(pair):
                design_application, design_rows = _design_application(
                    fold,
                    contrast=contrast,
                    role=role,
                )
                if receiver_support.status.value == "observed":
                    if pair_binding is None:
                        raise _source_mismatch(
                            "Supported receiver lacks its directional binding",
                            field_name="directional_response_bindings",
                        )
                    (
                        functional,
                        application,
                        common_binding,
                        model,
                        incremental_app,
                    ) = _parent_chain(
                        fold,
                        contrast_name=contrast.name,
                        contrast=contrast,
                        receiver=receiver,
                        directional=pair_binding,
                        role=role,
                    )
                    parts.append(
                        _channel_rows(
                            artifacts=artifacts,
                            pair=pair,
                            binding=pair_binding,
                            functional=functional,
                            application=application,
                            common_binding=common_binding,
                            model=model,
                            incremental_app=incremental_app,
                            receiver_support=receiver_support,
                            design_application=design_application,
                            design_rows=design_rows,
                            universe=universe,
                            role=role,
                            channel=channel,
                            contrast=contrast,
                        )
                    )
                    fitted_parent_values: dict[str, object] = {
                        "directional_binding_id": pair_binding.binding_id,
                        "directional_status": pair_binding.status,
                        "directional_reason_code": pair_binding.reason_code,
                        "family_common_functional_id": (
                            functional.family_common_functional_id
                        ),
                        "family_common_application_id": application.application_id,
                        "family_common_binding_id": common_binding.binding_id,
                        "family_common_member_scores_digest": (
                            application.member_scores_digest
                        ),
                        "family_common_family_scores_digest": (
                            application.family_scores_digest
                        ),
                        "edge_evidence_digest": common_binding.edge_evidence_digest,
                        "incremental_training_artifact_id": model.training_artifact_id,
                        "incremental_application_id": incremental_app.application_id,
                    }
                    contrast_manifest_id = functional.contrast_manifest_id
                else:
                    if (
                        receiver_support.reason_code != _ABSENT_RECEIVER_REASON
                        or pair_binding is not None
                    ):
                        raise _source_mismatch(
                            "Training-absent receiver has invalid directional lineage",
                            field_name="receiver_training_support",
                        )
                    parts.append(
                        _absent_receiver_channel_rows(
                            artifacts=artifacts,
                            pair=pair,
                            fold=fold,
                            receiver_support=receiver_support,
                            design_application=design_application,
                            design_rows=design_rows,
                            universe=universe,
                            role=role,
                            channel=channel,
                            contrast=contrast,
                        )
                    )
                    fitted_parent_values = {
                        "directional_binding_id": None,
                        "directional_status": None,
                        "directional_reason_code": None,
                        "family_common_functional_id": None,
                        "family_common_application_id": None,
                        "family_common_binding_id": None,
                        "family_common_member_scores_digest": None,
                        "family_common_family_scores_digest": None,
                        "edge_evidence_digest": None,
                        "incremental_training_artifact_id": None,
                        "incremental_application_id": None,
                    }
                    contrast_manifest_id = stable_id(
                        "contrast_manifest", contrast.to_dict()
                    )
                parent_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "receiver": receiver,
                        "receiver_training_support_id": (
                            receiver_support.support_record_id
                        ),
                        "receiver_training_support_status": (
                            receiver_support.status.value
                        ),
                        "receiver_training_support_reason_code": (
                            receiver_support.reason_code
                        ),
                        "directional_lr_hypothesis_universe_id": (universe.universe_id),
                        "receiver_family_lr_opportunity_axis_id": (
                            universe.opportunity_axis_id
                        ),
                        "design_application_id": design_application.application_id,
                        "channel_role": role,
                        "channel": channel,
                        "contrast_name": contrast.name,
                        "contrast_manifest_id": contrast_manifest_id,
                        **fitted_parent_values,
                    }
                )
    if not parts:
        raise _source_mismatch(
            "Directional LR source grid is empty", field_name="scores"
        )
    table = pd.concat(parts, ignore_index=True)
    table = table.loc[:, list(DIRECTIONAL_INTEGRATED_LR_COLUMNS)]
    table = table.sort_values(list(_ROW_KEY), kind="stable", ignore_index=True)
    return table, tuple(parent_rows)


def _build_table(
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
) -> tuple[pd.DataFrame, tuple[dict[str, object], ...]]:
    table, parent_rows = _materialize_table(artifacts, pair)
    _validate_table(table, artifacts=artifacts, pair=pair)
    return table, parent_rows


def _frozen_grid_contract(
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
) -> tuple[
    set[tuple[str, ...]],
    dict[tuple[str, str], Any],
    dict[tuple[str, str], tuple[str, str, str]],
    dict[tuple[str, str, str], dict[str, str]],
]:
    universe = _lr_hypothesis_universe(artifacts)
    expected_keys: set[tuple[str, ...]] = set()
    supports: dict[tuple[str, str], Any] = {}
    designs: dict[tuple[str, str], tuple[str, str, str]] = {}
    static_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for fold in artifacts.folds:
        for support in fold.receiver_training_support:
            supports[(fold.fold_id, support.receiver_id)] = support
            for opportunity in _frozen_lr_opportunities(
                universe,
                receiver=support.receiver_id,
            ):
                static_rows[
                    (
                        support.receiver_id,
                        opportunity["interaction_id"],
                        opportunity["mode"],
                    )
                ] = opportunity
        for role, _channel, contrast in _channel_specs(pair):
            design_application, design_rows = _design_application(
                fold,
                contrast=contrast,
                role=role,
            )
            designs[(fold.fold_id, role)] = (
                design_application.application_id,
                contrast.name,
                str(stable_id("contrast_manifest", contrast.to_dict())),
            )
            for support in fold.receiver_training_support:
                for opportunity in _frozen_lr_opportunities(
                    universe,
                    receiver=support.receiver_id,
                ):
                    expected_keys.update(
                        {
                            (
                                fold.fold_id,
                                support.receiver_id,
                                role,
                                sample_id,
                                subject_id,
                                context_id,
                                opportunity["family_id"],
                                opportunity["driver_id"],
                                opportunity["interaction_id"],
                                opportunity["mode"],
                            )
                            for sample_id, subject_id, context_id in design_rows
                        }
                    )
    return expected_keys, supports, designs, static_rows


def _validate_table(
    table: pd.DataFrame,
    *,
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
) -> None:
    universe = _lr_hypothesis_universe(artifacts)
    if tuple(table.columns) != DIRECTIONAL_INTEGRATED_LR_COLUMNS:
        raise ValueError("directional integrated LR columns changed")
    if table.empty:
        raise ValueError("directional integrated LR table cannot be empty")
    if table.duplicated(list(_ROW_KEY)).any():
        raise ValueError("directional integrated LR table contains duplicate keys")
    expected_keys, supports, designs, static_rows = _frozen_grid_contract(
        artifacts,
        pair,
    )
    observed_keys = {
        _key_tuple(record, _ROW_KEY) for record in table.to_dict(orient="records")
    }
    if observed_keys != expected_keys:
        raise ValueError(
            "directional LR table differs from the exact frozen source grid"
        )
    for column, expected in (
        ("crossfit_id", artifacts.crossfit_id),
        ("crossfit_spec_id", artifacts.spec.spec_id),
        ("pair_spec_id", pair.pair_spec_id),
        ("directional_lr_hypothesis_universe_id", universe.universe_id),
        ("score_version", DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION),
        ("semantics", DIRECTIONAL_INTEGRATED_LR_SEMANTICS),
        ("analysis_track", DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK),
    ):
        if set(table[column].astype(str)) != {expected}:
            raise ValueError(f"directional LR {column} changed lineage")
    if set(table["channel_role"].astype(str)) != {"forward", "reverse"}:
        raise ValueError("directional LR table must contain both channels")
    if set(table["channel"].astype(str)) != {
        pair.forward_channel_name,
        pair.reverse_channel_name,
    }:
        raise ValueError("directional LR channel names changed")
    forward = table.loc[table["channel_role"].eq("forward")]
    reverse = table.loc[table["channel_role"].eq("reverse")]
    if set(forward["channel"].astype(str)) != {pair.forward_channel_name}:
        raise ValueError("directional LR forward role has the wrong channel")
    if set(reverse["channel"].astype(str)) != {pair.reverse_channel_name}:
        raise ValueError("directional LR reverse role has the wrong channel")
    for column in (
        "experimental",
        "source_agnostic",
        "cross_channel_comparable",
        "paired_score_comparison_allowed",
        "active_inhibition_allowed",
        "supports_active_inhibition_claim",
        "formal_inference_allowed",
    ):
        values = table[column].tolist()
        if any(type(value) is not bool for value in values):
            raise ValueError(f"directional LR {column} must be boolean")
    if (
        not bool(table["experimental"].all())
        or bool(table["source_agnostic"].any())
        or bool(table["cross_channel_comparable"].any())
        or bool(table["paired_score_comparison_allowed"].any())
        or bool(table["active_inhibition_allowed"].any())
        or bool(table["supports_active_inhibition_claim"].any())
        or bool(table["formal_inference_allowed"].any())
    ):
        raise ValueError("directional LR scope flags are invalid")
    for row in table.to_dict(orient="records"):
        fold_id = str(row["fold_id"])
        receiver = str(row["receiver"])
        role = str(row["channel_role"])
        try:
            support = supports[(fold_id, receiver)]
            design_id, contrast_name, contrast_manifest_id = designs[(fold_id, role)]
            static = static_rows[
                (receiver, str(row["interaction_id"]), str(row["mode"]))
            ]
        except KeyError as error:  # pragma: no cover - exact-grid guard
            raise ValueError("directional LR row is outside the frozen grid") from error
        if (
            row["receiver_training_support_id"] != support.support_record_id
            or row["receiver_training_support_status"] != support.status.value
            or _optional_text(row["receiver_training_support_reason_code"])
            != support.reason_code
            or row["design_application_id"] != design_id
            or row["contrast_name"] != contrast_name
            or row["contrast_manifest_id"] != contrast_manifest_id
            or str(row["family_id"]) != static["family_id"]
            or str(row["driver_id"]) != static["driver_id"]
            or str(row["molecular_lr_equivalence_id"])
            != static["molecular_lr_equivalence_id"]
            or row["receiver_family_lr_membership_id"]
            != static["receiver_family_lr_membership_id"]
            or row["receiver_family_lr_hypothesis_id"]
            != static["receiver_family_lr_hypothesis_id"]
            or row["receiver_family_lr_opportunity_id"]
            != static["receiver_family_lr_opportunity_id"]
        ):
            raise ValueError(
                "directional LR static, support, or design lineage changed"
            )
        support_status = str(row["receiver_training_support_status"])
        support_reason = _optional_text(row["receiver_training_support_reason_code"])
        source_status = _optional_text(row["source_status"])
        source_reason = _optional_text(row["source_reason_code"])
        fitted_parents = {
            column: _optional_text(row[column])
            for column in _FITTED_SCORE_PARENT_COLUMNS
        }
        if support_status == "not_estimable":
            if (
                support_reason != _ABSENT_RECEIVER_REASON
                or source_status is not None
                or source_reason is not None
                or any(value is not None for value in fitted_parents.values())
                or row["status"] != "not_estimable"
                or _optional_text(row["reason_code"]) != _ABSENT_RECEIVER_REASON
                or _optional_float(
                    row["integrated_lr_score"], field_name="integrated_lr_score"
                )
                is not None
                or _optional_float(
                    row["family_core_strength"], field_name="family_core_strength"
                )
                is not None
                or _optional_float(
                    row["within_family_lr_weight"],
                    field_name="within_family_lr_weight",
                )
                is not None
            ):
                raise ValueError(
                    "training-absent directional LR row is not exact typed NE"
                )
        elif support_status == "observed":
            if (
                support_reason is not None
                or source_status not in _STATUS_VALUES
                or any(value is None for value in fitted_parents.values())
            ):
                raise ValueError(
                    "supported directional LR row lacks source or fitted lineage"
                )
            if (source_status == "observed" and source_reason is not None) or (
                source_status in {"structural_zero", "not_estimable"}
                and source_reason is None
            ):
                raise ValueError("directional LR source status/reason is invalid")
        else:
            raise ValueError("directional LR receiver support status is invalid")
        for column in (
            "molecular_lr_equivalence_id",
            "receiver_training_support_id",
            "design_application_id",
            "receiver_family_lr_membership_id",
            "receiver_family_lr_hypothesis_id",
            "receiver_family_lr_opportunity_id",
        ):
            _required_name(row[column], field_name=column)
    for value, status, reason in table.loc[
        :, ["integrated_lr_score", "status", "reason_code"]
    ].itertuples(index=False, name=None):
        missing = value is None or bool(pd.isna(cast(Any, value)))
        status_text = str(status)
        reason_text = _optional_text(reason)
        if status_text not in _STATUS_VALUES:
            raise ValueError("directional LR table has an unsupported status")
        if status_text == "observed" and (
            missing or reason_text is not None or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("observed directional LR rows are invalid")
        if status_text == "structural_zero" and (
            missing or float(value) != 0.0 or reason_text is None
        ):
            raise ValueError("structural-zero directional LR rows are invalid")
        if status_text == "not_estimable" and (not missing or reason_text is None):
            raise ValueError("not-estimable directional LR rows are invalid")
    # Both channels must describe exactly the same frozen LR row universe.
    forward_keys = {
        tuple(str(row[column]) for column in _CHANNEL_KEY)
        for row in forward.to_dict(orient="records")
    }
    reverse_keys = {
        tuple(str(row[column]) for column in _CHANNEL_KEY)
        for row in reverse.to_dict(orient="records")
    }
    if forward_keys != reverse_keys:
        raise ValueError("directional LR channels do not share an exact key grid")
    if set(forward["fold_id"].astype(str)) != set(reverse["fold_id"].astype(str)):
        raise ValueError("directional LR channels do not share fold coverage")
    # Only real source allocations, never synthetic absent-receiver rows, can
    # participate in family conservation.
    for _group_key, group in table.groupby(
        [
            "fold_id",
            "receiver",
            "channel_role",
            "sample_id",
            "subject_id",
            "context_id",
            "family_id",
            "mode",
        ],
        observed=True,
        sort=False,
    ):
        source_presence = [
            _optional_text(value) is not None for value in group["source_status"]
        ]
        if not any(source_presence):
            continue
        if not all(source_presence):
            raise ValueError(
                "directional LR allocation mixes source and synthetic rows"
            )
        if group["status"].eq("not_estimable").any():
            if not group["status"].eq("not_estimable").all():
                raise ValueError("directional LR allocation mixes observed and NE rows")
            continue
        scores = group["integrated_lr_score"].astype(float)
        core = group["family_core_strength"]
        if core.isna().any() or not np.isfinite(core.astype(float)).all():
            raise ValueError("directional LR family core is missing")
        if not np.allclose(core.astype(float), core.iloc[0]):
            raise ValueError("directional LR family core changed within a group")
        if not math.isclose(
            float(scores.sum()),
            float(core.iloc[0]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("directional LR members do not conserve family core")


def _build_opportunity_registry(
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
    table: pd.DataFrame,
) -> pd.DataFrame:
    universe = _lr_hypothesis_universe(artifacts)
    score_counts = _score_row_counts(table)
    rows: list[dict[str, object]] = []
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        directional = _directional_bindings(fold, pair)
        support_records = tuple(fold.receiver_training_support)
        if tuple(record.receiver_id for record in support_records) != (
            artifacts.receiver_universe.receiver_ids
        ):
            raise _source_mismatch(
                "Fold receiver support does not retain the run receiver axis",
                field_name="receiver_training_support",
            )
        for support in support_records:
            support._require_intact()
            receiver = support.receiver_id
            binding = directional.get(receiver)
            forward_count = score_counts.get((fold.fold_id, receiver, "forward"), 0)
            reverse_count = score_counts.get((fold.fold_id, receiver, "reverse"), 0)
            channel_parent_ids: dict[str, object]
            if support.status == "observed":
                if binding is None:
                    raise _source_mismatch(
                        "Supported directional opportunity lacks its binding",
                        field_name="directional_response_bindings",
                    )
                status = binding.status
                reason_code = binding.reason_code
                grid_status = "produced"
                grid_reason_code = None
                channel_parent_ids = _opportunity_parent_ids(
                    table,
                    fold_id=fold.fold_id,
                    receiver=receiver,
                    binding=binding,
                )
            else:
                if binding is not None:
                    raise _source_mismatch(
                        "Training-absent receiver cannot have a directional binding",
                        field_name="directional_response_bindings",
                    )
                status = "not_estimable"
                reason_code = support.reason_code
                grid_status = "produced"
                grid_reason_code = None
                channel_parent_ids = {
                    column: None for column in _OPPORTUNITY_PARENT_COLUMNS
                }
            rows.append(
                {
                    "crossfit_id": artifacts.crossfit_id,
                    "crossfit_spec_id": artifacts.spec.spec_id,
                    "pair_spec_id": pair.pair_spec_id,
                    "directional_lr_hypothesis_universe_id": universe.universe_id,
                    "receiver_family_lr_opportunity_axis_id": (
                        universe.opportunity_axis_id
                    ),
                    "receiver_universe_id": (artifacts.receiver_universe.universe_id),
                    "receiver_axis_id": artifacts.receiver_universe.receiver_axis_id,
                    "fold_id": fold.fold_id,
                    "receiver": receiver,
                    "receiver_training_support_id": support.support_record_id,
                    "receiver_training_support_status": support.status.value,
                    "receiver_training_support_reason_code": support.reason_code,
                    "directional_binding_id": (
                        None if binding is None else binding.binding_id
                    ),
                    **channel_parent_ids,
                    "forward_channel": pair.forward_channel_name,
                    "reverse_channel": pair.reverse_channel_name,
                    "status": status,
                    "reason_code": reason_code,
                    "lr_hypothesis_grid_status": grid_status,
                    "lr_hypothesis_grid_reason_code": grid_reason_code,
                    "n_forward_score_rows": forward_count,
                    "n_reverse_score_rows": reverse_count,
                    "paired_score_comparison_allowed": False,
                    "active_inhibition_allowed": False,
                    "supports_active_inhibition_claim": False,
                    "formal_inference_allowed": False,
                }
            )
    registry = pd.DataFrame(
        rows,
        columns=DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS,
    ).sort_values(["fold_id", "receiver"], kind="stable", ignore_index=True)
    _validate_opportunity_registry(
        registry,
        artifacts=artifacts,
        pair=pair,
        table=table,
    )
    return registry


def _validate_opportunity_registry(
    registry: pd.DataFrame,
    *,
    artifacts: CrossFitArtifacts,
    pair: DirectionalContrastPairSpec,
    table: pd.DataFrame,
) -> None:
    universe = _lr_hypothesis_universe(artifacts)
    if tuple(registry.columns) != DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS:
        raise ValueError("directional LR opportunity columns changed")
    if registry.empty or registry.duplicated(["fold_id", "receiver"]).any():
        raise ValueError("directional LR opportunities must be non-empty and unique")
    expected_keys = {
        (fold.fold_id, receiver)
        for fold in artifacts.folds
        for receiver in artifacts.receiver_universe.receiver_ids
    }
    observed_keys = set(
        registry.loc[:, ["fold_id", "receiver"]].itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise ValueError(
            "directional LR opportunities do not cover the run receiver axis"
        )
    for column, expected in (
        ("crossfit_id", artifacts.crossfit_id),
        ("crossfit_spec_id", artifacts.spec.spec_id),
        ("pair_spec_id", pair.pair_spec_id),
        ("directional_lr_hypothesis_universe_id", universe.universe_id),
        (
            "receiver_family_lr_opportunity_axis_id",
            universe.opportunity_axis_id,
        ),
        ("receiver_universe_id", artifacts.receiver_universe.universe_id),
        ("receiver_axis_id", artifacts.receiver_universe.receiver_axis_id),
        ("forward_channel", pair.forward_channel_name),
        ("reverse_channel", pair.reverse_channel_name),
    ):
        if set(registry[column].astype(str)) != {expected}:
            raise ValueError(f"directional LR opportunity {column} changed")
    actual_counts = _score_row_counts(table)
    fold_map = _fold_map(artifacts)
    for row in registry.to_dict(orient="records"):
        fold_id = str(row["fold_id"])
        receiver = str(row["receiver"])
        fold = fold_map[fold_id]
        support = next(
            record
            for record in fold.receiver_training_support
            if record.receiver_id == receiver
        )
        directional = _directional_bindings(fold, pair)
        binding = directional.get(receiver)
        reason = _optional_text(row["reason_code"])
        support_reason = _optional_text(row["receiver_training_support_reason_code"])
        grid_reason = _optional_text(row["lr_hypothesis_grid_reason_code"])
        binding_id = _optional_text(row["directional_binding_id"])
        forward_count = int(cast(Any, row["n_forward_score_rows"]))
        reverse_count = int(cast(Any, row["n_reverse_score_rows"]))
        expected_forward_count = actual_counts.get((fold_id, receiver, "forward"), 0)
        expected_reverse_count = actual_counts.get((fold_id, receiver, "reverse"), 0)
        observed_parent_ids = {
            column: _optional_text(row[column])
            for column in _OPPORTUNITY_PARENT_COLUMNS
        }
        if (
            row["receiver_training_support_id"] != support.support_record_id
            or row["receiver_training_support_status"] != support.status.value
            or support_reason != support.reason_code
            or forward_count != expected_forward_count
            or reverse_count != expected_reverse_count
        ):
            raise ValueError(
                "directional LR opportunity support or score lineage changed"
            )
        if support.status == "observed":
            if binding is None:  # pragma: no cover - source coverage invariant
                raise ValueError("supported directional opportunity lacks binding")
            expected_parent_ids = _opportunity_parent_ids(
                table,
                fold_id=fold_id,
                receiver=receiver,
                binding=binding,
            )
            if (
                binding_id != binding.binding_id
                or observed_parent_ids != expected_parent_ids
                or row["status"] != binding.status
                or reason != binding.reason_code
                or row["lr_hypothesis_grid_status"] != "produced"
                or grid_reason is not None
                or forward_count <= 0
                or forward_count != reverse_count
            ):
                raise ValueError(
                    "supported directional LR opportunity lacks a symmetric "
                    "two-channel parent grid"
                )
        elif (
            support.status != "not_estimable"
            or support.reason_code != _ABSENT_RECEIVER_REASON
            or binding is not None
            or binding_id is not None
            or any(value is not None for value in observed_parent_ids.values())
            or row["status"] != "not_estimable"
            or reason != _ABSENT_RECEIVER_REASON
            or row["lr_hypothesis_grid_status"] != "produced"
            or grid_reason is not None
            or forward_count <= 0
            or forward_count != reverse_count
        ):
            raise ValueError(
                "training-absent directional LR opportunity is not typed NE"
            )
    for column in (
        "paired_score_comparison_allowed",
        "active_inhibition_allowed",
        "supports_active_inhibition_claim",
        "formal_inference_allowed",
    ):
        values = registry[column].tolist()
        if any(type(value) is not bool or value for value in values):
            raise ValueError(f"directional LR opportunity {column} must remain false")


@dataclass(frozen=True, slots=True, init=False)
class DirectionalIntegratedLRCollection:
    """Producer-owned, non-additive forward/reverse LR score views."""

    collection_id: str
    crossfit_id: str
    crossfit_spec_id: str
    pair_spec_id: str
    directional_lr_hypothesis_universe_id: str
    molecular_lr_equivalence_universe_id: str
    molecular_lr_axis_id: str
    receiver_family_lr_opportunity_axis_id: str
    receiver_universe_id: str
    receiver_axis_id: str
    channels: tuple[str, str]
    table_digest: str
    row_count: int
    status_counts: tuple[tuple[str, int], ...]
    opportunity_registry_digest: str
    opportunity_count: int
    opportunity_status_counts: tuple[tuple[str, int], ...]
    score_grid_complete_on_run_receiver_axis: bool
    experimental: bool
    source_agnostic: bool
    cross_channel_comparable: bool
    paired_score_comparison_allowed: bool
    active_inhibition_allowed: bool
    supports_active_inhibition_claim: bool
    formal_inference_allowed: bool
    parent_bindings: tuple[dict[str, object], ...]
    _scores: pd.DataFrame = field(repr=False)
    _opportunity_registry: pd.DataFrame = field(repr=False)
    _source_artifacts: CrossFitArtifacts = field(repr=False)
    _pair_spec: DirectionalContrastPairSpec = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "DirectionalIntegratedLRCollection is producer-owned; use "
            "build_directional_integrated_lr_scores()"
        )

    @classmethod
    def _from_table(
        cls,
        artifacts: CrossFitArtifacts,
        pair: DirectionalContrastPairSpec,
        table: pd.DataFrame,
        parent_bindings: tuple[dict[str, object], ...],
    ) -> DirectionalIntegratedLRCollection:
        self = object.__new__(cls)
        universe = _lr_hypothesis_universe(artifacts)
        opportunities = _build_opportunity_registry(artifacts, pair, table)
        values: dict[str, object] = {
            "crossfit_id": artifacts.crossfit_id,
            "crossfit_spec_id": artifacts.spec.spec_id,
            "pair_spec_id": pair.pair_spec_id,
            "directional_lr_hypothesis_universe_id": universe.universe_id,
            "molecular_lr_equivalence_universe_id": (
                universe.molecular_lr_equivalence_universe_id
            ),
            "molecular_lr_axis_id": universe.molecular_lr_axis_id,
            "receiver_family_lr_opportunity_axis_id": (universe.opportunity_axis_id),
            "receiver_universe_id": artifacts.receiver_universe.universe_id,
            "receiver_axis_id": artifacts.receiver_universe.receiver_axis_id,
            "channels": (pair.forward_channel_name, pair.reverse_channel_name),
            "table_digest": _table_digest(table),
            "row_count": len(table),
            "status_counts": _status_counts(table),
            "opportunity_registry_digest": _table_digest(opportunities),
            "opportunity_count": len(opportunities),
            "opportunity_status_counts": _opportunity_status_counts(opportunities),
            "score_grid_complete_on_run_receiver_axis": bool(
                opportunities["lr_hypothesis_grid_status"].eq("produced").all()
            ),
            "experimental": True,
            "source_agnostic": False,
            "cross_channel_comparable": False,
            "paired_score_comparison_allowed": False,
            "active_inhibition_allowed": False,
            "supports_active_inhibition_claim": False,
            "formal_inference_allowed": False,
            "parent_bindings": tuple(
                sorted(
                    parent_bindings,
                    key=lambda item: (
                        str(item["fold_id"]),
                        str(item["receiver"]),
                        str(item["channel_role"]),
                    ),
                )
            ),
            "_scores": table.copy(deep=True),
            "_opportunity_registry": opportunities.copy(deep=True),
            "_source_artifacts": artifacts,
            "_pair_spec": pair,
            "_producer_marker": _COLLECTION_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "collection_id",
            stable_id(
                "directional_integrated_lr_collection",
                self._identity_payload(),
                schema_version=_IDENTITY_SCHEMA_VERSION,
            ),
        )
        self._require_intact()
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "active_inhibition_allowed": self.active_inhibition_allowed,
            "channels": list(self.channels),
            "cross_channel_comparable": self.cross_channel_comparable,
            "crossfit_id": self.crossfit_id,
            "crossfit_spec_id": self.crossfit_spec_id,
            "directional_lr_hypothesis_universe_id": (
                self.directional_lr_hypothesis_universe_id
            ),
            "molecular_lr_equivalence_universe_id": (
                self.molecular_lr_equivalence_universe_id
            ),
            "molecular_lr_axis_id": self.molecular_lr_axis_id,
            "experimental": self.experimental,
            "formal_inference_allowed": self.formal_inference_allowed,
            "opportunity_count": self.opportunity_count,
            "opportunity_registry_digest": self.opportunity_registry_digest,
            "opportunity_status_counts": [
                list(item) for item in self.opportunity_status_counts
            ],
            "pair_spec_id": self.pair_spec_id,
            "parent_bindings": list(self.parent_bindings),
            "paired_score_comparison_allowed": self.paired_score_comparison_allowed,
            "receiver_axis_id": self.receiver_axis_id,
            "receiver_family_lr_opportunity_axis_id": (
                self.receiver_family_lr_opportunity_axis_id
            ),
            "receiver_universe_id": self.receiver_universe_id,
            "row_count": self.row_count,
            "score_grid_complete_on_run_receiver_axis": (
                self.score_grid_complete_on_run_receiver_axis
            ),
            "source_agnostic": self.source_agnostic,
            "status_counts": [list(item) for item in self.status_counts],
            "supports_active_inhibition_claim": self.supports_active_inhibition_claim,
            "table_digest": self.table_digest,
            "score_version": DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION,
            "semantics": DIRECTIONAL_INTEGRATED_LR_SEMANTICS,
        }

    def _require_intact(self) -> None:
        try:
            self._source_artifacts._require_intact()
            self._pair_spec._require_intact()
            universe = _lr_hypothesis_universe(self._source_artifacts)
            if self.pair_spec_id != self._pair_spec.pair_spec_id:
                raise ValueError("directional pair identity changed")
            expected, parents = _build_table(self._source_artifacts, self._pair_spec)
            expected_opportunities = _build_opportunity_registry(
                self._source_artifacts,
                self._pair_spec,
                expected,
            )
            _validate_table(
                self._scores,
                artifacts=self._source_artifacts,
                pair=self._pair_spec,
            )
            _validate_opportunity_registry(
                self._opportunity_registry,
                artifacts=self._source_artifacts,
                pair=self._pair_spec,
                table=self._scores,
            )
            expected_parent_tuple = tuple(
                sorted(
                    parents,
                    key=lambda item: (
                        str(item["fold_id"]),
                        str(item["receiver"]),
                        str(item["channel_role"]),
                    ),
                )
            )
            valid = (
                self._producer_marker == _COLLECTION_MARKER
                and self.crossfit_id == self._source_artifacts.crossfit_id
                and self.crossfit_spec_id == self._source_artifacts.spec.spec_id
                and self.directional_lr_hypothesis_universe_id == universe.universe_id
                and self.molecular_lr_equivalence_universe_id
                == universe.molecular_lr_equivalence_universe_id
                and self.molecular_lr_axis_id == universe.molecular_lr_axis_id
                and self.receiver_family_lr_opportunity_axis_id
                == universe.opportunity_axis_id
                and self.receiver_universe_id
                == self._source_artifacts.receiver_universe.universe_id
                and self.receiver_axis_id
                == self._source_artifacts.receiver_universe.receiver_axis_id
                and self.channels
                == (
                    self._pair_spec.forward_channel_name,
                    self._pair_spec.reverse_channel_name,
                )
                and self.parent_bindings == expected_parent_tuple
                and self.table_digest == _table_digest(self._scores)
                and self.table_digest == _table_digest(expected)
                and self._scores.equals(expected)
                and self.row_count == len(self._scores) == len(expected)
                and self.status_counts == _status_counts(self._scores)
                and self.opportunity_registry_digest
                == _table_digest(self._opportunity_registry)
                and self.opportunity_registry_digest
                == _table_digest(expected_opportunities)
                and self._opportunity_registry.equals(expected_opportunities)
                and self.opportunity_count
                == len(self._opportunity_registry)
                == len(expected_opportunities)
                and self.opportunity_status_counts
                == _opportunity_status_counts(self._opportunity_registry)
                and self.score_grid_complete_on_run_receiver_axis
                is bool(
                    self._opportunity_registry["lr_hypothesis_grid_status"]
                    .eq("produced")
                    .all()
                )
                and self.experimental is True
                and self.source_agnostic is False
                and self.cross_channel_comparable is False
                and self.paired_score_comparison_allowed is False
                and self.active_inhibition_allowed is False
                and self.supports_active_inhibition_claim is False
                and self.formal_inference_allowed is False
                and self.collection_id
                == stable_id(
                    "directional_integrated_lr_collection",
                    self._identity_payload(),
                    schema_version=_IDENTITY_SCHEMA_VERSION,
                )
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Directional integrated LR collection failed integrity validation",
                code="directional_integrated_lr_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild it from one intact directional cross-fit",
            ) from error
        if not valid:
            raise ContractError(
                "Directional integrated LR collection failed integrity validation",
                code="directional_integrated_lr_collection_integrity_violation",
                field="collection_id",
                remediation="Rebuild it from one intact directional cross-fit",
            )

    @property
    def scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._scores.copy(deep=True)

    @property
    def opportunity_registry(self) -> pd.DataFrame:
        """Return complete fold-by-run-receiver pair opportunities."""

        self._require_intact()
        return self._opportunity_registry.copy(deep=True)

    def to_manifest(self) -> dict[str, object]:
        self._require_intact()
        return {
            "collection_id": self.collection_id,
            "schema_version": _SCHEMA_VERSION,
            "producer": _PRODUCER,
            **self._identity_payload(),
            "grain": (
                "outer_fold_x_pair_x_channel_x_sample_x_receiver_x_family_x_lr_x_mode"
            ),
            "opportunity_registry_grain": "outer_fold_x_pair_x_run_receiver",
            "opportunity_registry_complete": True,
            "score_grid_scope": _FULL_GRID_SCOPE,
            "status": (
                "complete_descriptive"
                if self.score_grid_complete_on_run_receiver_axis
                else "complete_descriptive_with_receiver_level_not_estimable"
            ),
            "comparability_scope": "within_channel_only",
            "combination_rule": "independent_contrast_views_not_additive_v1",
            "p_value": None,
            "q_value": None,
            "comm_probability": None,
        }


def build_directional_integrated_lr_scores(
    artifacts: CrossFitArtifacts,
    *,
    pair_spec_id: str,
) -> DirectionalIntegratedLRCollection:
    """Build independent forward/reverse LR views without refitting."""

    if type(artifacts) is not CrossFitArtifacts:
        raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
    artifacts._require_intact()
    pair = _pair_for_id(artifacts, pair_spec_id)
    if artifacts.spec.penalty_tuning_spec is None:
        raise _source_mismatch(
            "Directional integrated LR views require family-common tuned parents",
            field_name="crossfit.spec.penalty_tuning_spec",
        )
    table, parents = _build_table(artifacts, pair)
    _validate_table(table, artifacts=artifacts, pair=pair)
    return DirectionalIntegratedLRCollection._from_table(
        artifacts,
        pair,
        table,
        parents,
    )


__all__ = [
    "DIRECTIONAL_INTEGRATED_LR_ANALYSIS_TRACK",
    "DIRECTIONAL_INTEGRATED_LR_COLUMNS",
    "DIRECTIONAL_INTEGRATED_LR_OPPORTUNITY_COLUMNS",
    "DIRECTIONAL_INTEGRATED_LR_SCORE_VERSION",
    "DIRECTIONAL_INTEGRATED_LR_SEMANTICS",
    "DirectionalIntegratedLRCollection",
    "build_directional_integrated_lr_scores",
]
