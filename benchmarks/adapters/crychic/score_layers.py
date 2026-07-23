"""Versioned multigroup communication score layers for benchmark adapters."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from crychic.scoring import DownstreamEvidencePolicy

SCORE_LAYER_SCHEMA_VERSION = "crychic-multigroup-score-layers-v2"
BOUNDED_DETECTION_EVIDENCE_VERSION = "bounded-detection-evidence-v1"
BOUNDED_DETECTION_MECHANISM_FLOOR = 0.75
BOUNDED_DETECTION_DOWNSTREAM_WEIGHT = 0.05

_BASE_KEYS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
)
_SENDER_KEYS = (*_BASE_KEYS, "sender")
_DOWNSTREAM_KEYS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "receiver",
    "subject_id",
    "family_id",
)
_RECEIVER_PROGRAM_KEYS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "mode",
)

SCORE_LAYER_VALUE_COLUMNS = (
    "mechanistic_lr_score",
    "mechanistic_assignment_weight",
    "mechanistic_sender_lr_score",
    "mechanistic_status",
    "mechanistic_reason_code",
    "downstream_loss_ratio_effect",
    "downstream_signed_support",
    "downstream_positive_support",
    "downstream_status",
    "downstream_reason_code",
    "downstream_modulated_sender_lr_score",
    "downstream_confirmed_sender_lr_score",
    "downstream_confirmed_status",
    "downstream_confirmed_reason_code",
    "receiver_program_score",
    "receiver_program_status",
    "receiver_program_reason_code",
    "bounded_detection_evidence_score",
    "bounded_detection_evidence_status",
    "bounded_detection_evidence_reason_code",
    "bounded_detection_evidence_version",
    "bounded_detection_evidence_formal_inference_allowed",
    "selected_score",
    "selected_score_policy",
    "selected_score_status",
    "selected_score_reason_code",
    "score_layer_schema_version",
)


def _require_columns(
    table: pd.DataFrame, required: Sequence[str], *, name: str
) -> None:
    missing = set(required).difference(table.columns)
    if missing:
        raise ValueError(f"{name} is missing score-layer columns: {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{name} must not be empty")


def _unit_series(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    invalid = values.notna() & numeric.isna()
    finite = numeric.dropna()
    if invalid.any() or (
        not finite.empty
        and ((finite < 0.0).any() or (finite > 1.0).any() or np.isinf(finite).any())
    ):
        raise ValueError(f"{field} must contain values in [0, 1] or missing")
    return numeric


def _finite_series(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    invalid = values.notna() & numeric.isna()
    if invalid.any() or np.isinf(numeric.dropna()).any():
        raise ValueError(f"{field} must contain finite values or missing")
    return numeric


def _modulate_vectorized(
    mechanism: pd.Series,
    support: pd.Series,
    *,
    modulation_strength: float,
    epsilon: float = 1e-8,
) -> pd.Series:
    strength = float(modulation_strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("modulation_strength must be finite and non-negative")
    result = mechanism.copy()
    estimable = mechanism.notna() & support.notna()
    boundary = estimable & (mechanism.eq(0.0) | mechanism.eq(1.0))
    interior = estimable & ~boundary
    if interior.any():
        m = mechanism.loc[interior].clip(epsilon, 1.0 - epsilon)
        d = support.loc[interior].clip(-1.0, 1.0)
        logit = np.log(m / (1.0 - m))
        result.loc[interior] = 1.0 / (1.0 + np.exp(-(logit + strength * d)))
    return result


def bounded_detection_evidence_score(
    mechanistic_lr: pd.Series,
    sender_assignment: pd.Series,
    receiver_program: pd.Series,
    *,
    mechanism_floor: float = BOUNDED_DETECTION_MECHANISM_FLOOR,
    downstream_weight: float = BOUNDED_DETECTION_DOWNSTREAM_WEIGHT,
) -> pd.Series:
    """Return an unsigned detection score with a soft mechanism penalty.

    Missing optional receiver-program evidence falls back to sender evidence.
    Missing mechanism or sender evidence remains missing. This quantity is a
    ranking diagnostic, not communication strength or probability.
    """

    floor = float(mechanism_floor)
    weight = float(downstream_weight)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("mechanism_floor must lie in [0, 1]")
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("downstream_weight must lie in [0, 1]")
    mechanism = _unit_series(mechanistic_lr, field="mechanistic_lr")
    sender = _unit_series(sender_assignment, field="sender_assignment")
    program = _unit_series(receiver_program, field="receiver_program")
    component = sender.copy()
    available = program.notna()
    component.loc[available] = (1.0 - weight) * sender.loc[
        available
    ] + weight * program.loc[available]
    result = (floor + (1.0 - floor) * mechanism) * component
    result.loc[mechanism.isna() | sender.isna()] = np.nan
    return result


def build_multigroup_score_layers(
    compact_scores: pd.DataFrame,
    sender_components: pd.DataFrame,
    lr_components: pd.DataFrame,
    downstream_effects: pd.DataFrame,
    *,
    receiver_program_components: pd.DataFrame | None = None,
    policy: DownstreamEvidencePolicy | str = DownstreamEvidencePolicy.ANNOTATE,
    modulation_strength: float = 1.0,
) -> pd.DataFrame:
    """Attach independent mechanism, downstream, hybrid, and strict score layers.

    The mechanism layer uses held-out LR availability and prior quality, then
    allocates that parent with the frozen train-derived sender weight.  It does
    not consume the ligand contrast gate or receiver downstream gain.
    """

    resolved_policy = DownstreamEvidencePolicy(policy)
    _require_columns(
        compact_scores,
        (*_SENDER_KEYS, "global_sender_lr_score", "status", "reason_code"),
        name="compact_scores",
    )
    _require_columns(
        sender_components,
        (*_SENDER_KEYS, "assignment_weight"),
        name="sender_components",
    )
    _require_columns(
        lr_components,
        (
            *_BASE_KEYS,
            "receptor_eligible",
            "availability",
            "prior_quality",
        ),
        name="lr_components",
    )
    _require_columns(
        downstream_effects,
        (*_DOWNSTREAM_KEYS, "differential_effect", "status", "reason_code"),
        name="downstream_effects",
    )
    if compact_scores.duplicated(list(_SENDER_KEYS)).any():
        raise ValueError("compact_scores keys must be unique")

    if receiver_program_components is not None:
        _require_columns(
            receiver_program_components,
            (
                *_RECEIVER_PROGRAM_KEYS,
                "component_scope",
                "component",
                "component_value",
                "status",
                "reason_code",
            ),
            name="receiver_program_components",
        )

    senders = sender_components.loc[
        sender_components["mode"].astype(str).eq("state"),
        [*_SENDER_KEYS, "assignment_weight"],
    ]
    if senders.duplicated(list(_SENDER_KEYS)).any():
        raise ValueError("sender component keys must be unique")
    parents = lr_components.loc[
        lr_components["mode"].astype(str).eq("state"),
        [
            *_BASE_KEYS,
            "receptor_eligible",
            "availability",
            "prior_quality",
        ],
    ]
    if parents.duplicated(list(_BASE_KEYS)).any():
        raise ValueError("LR component keys must be unique")
    downstream = downstream_effects.loc[
        :,
        [*_DOWNSTREAM_KEYS, "differential_effect", "status", "reason_code"],
    ].rename(
        columns={
            "status": "_downstream_source_status",
            "reason_code": "_downstream_source_reason",
        }
    )
    if downstream.duplicated(list(_DOWNSTREAM_KEYS)).any():
        raise ValueError("downstream effect keys must be unique")

    if receiver_program_components is None:
        receiver_program = compact_scores.loc[:, list(_RECEIVER_PROGRAM_KEYS)].copy()
        receiver_program = receiver_program.drop_duplicates(
            list(_RECEIVER_PROGRAM_KEYS), ignore_index=True
        )
        receiver_program["component_value"] = np.nan
        receiver_program["_receiver_program_source_status"] = "not_estimable"
        receiver_program["_receiver_program_source_reason"] = (
            "receiver_program_component_not_supplied"
        )
    else:
        receiver_program = receiver_program_components.loc[
            receiver_program_components["component_scope"].astype(str).eq("family")
            & receiver_program_components["component"]
            .astype(str)
            .eq("receiver_program_score"),
            [
                *_RECEIVER_PROGRAM_KEYS,
                "component_value",
                "status",
                "reason_code",
            ],
        ].rename(
            columns={
                "status": "_receiver_program_source_status",
                "reason_code": "_receiver_program_source_reason",
            }
        )
        if receiver_program.empty:
            raise ValueError(
                "receiver_program_components has no family receiver_program_score rows"
            )
        if receiver_program.duplicated(list(_RECEIVER_PROGRAM_KEYS)).any():
            raise ValueError("receiver program component keys must be unique")

    sender_values = compact_scores.loc[:, list(_SENDER_KEYS)].merge(
        senders,
        on=list(_SENDER_KEYS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    parent_values = compact_scores.loc[:, list(_BASE_KEYS)].merge(
        parents,
        on=list(_BASE_KEYS),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    downstream_values = compact_scores.loc[:, list(_DOWNSTREAM_KEYS)].merge(
        downstream,
        on=list(_DOWNSTREAM_KEYS),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    receiver_program_values = compact_scores.loc[:, list(_RECEIVER_PROGRAM_KEYS)].merge(
        receiver_program,
        on=list(_RECEIVER_PROGRAM_KEYS),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if not (
        len(sender_values)
        == len(parent_values)
        == len(downstream_values)
        == len(receiver_program_values)
        == len(compact_scores)
    ):
        raise RuntimeError("score-layer joins changed sender row coverage")

    availability = _unit_series(parent_values["availability"], field="availability")
    prior_quality = _unit_series(parent_values["prior_quality"], field="prior_quality")
    assignment = _unit_series(
        sender_values["assignment_weight"], field="assignment_weight"
    )
    receptor = parent_values["receptor_eligible"]
    invalid_receptor = receptor.notna() & ~receptor.isin((True, False))
    if invalid_receptor.any():
        raise ValueError("receptor_eligible must contain booleans or missing")
    receptor_eligible = receptor.astype("boolean")

    parent_zero = (
        receptor_eligible.eq(False).fillna(False)
        | availability.eq(0.0)
        | prior_quality.eq(0.0)
    )
    parent_missing = (
        receptor_eligible.isna() | availability.isna() | prior_quality.isna()
    ) & ~parent_zero
    mechanistic_lr = availability * prior_quality
    mechanistic_lr.loc[parent_zero] = 0.0
    mechanistic_lr.loc[parent_missing] = np.nan
    mechanistic_sender = mechanistic_lr * assignment
    mechanistic_sender.loc[parent_zero] = 0.0

    work_index = sender_values.index
    mechanistic_status = pd.Series("observed", index=work_index, dtype=object)
    mechanistic_reason = pd.Series(None, index=work_index, dtype=object)
    mechanistic_status.loc[parent_zero] = "structural_zero"
    mechanistic_reason.loc[parent_zero] = "mechanistic_parent_structural_zero"
    mechanistic_status.loc[parent_missing] = "not_estimable"
    mechanistic_reason.loc[parent_missing] = "mechanistic_parent_not_estimable"
    sender_missing = assignment.isna() & ~parent_zero & ~parent_missing
    mechanistic_status.loc[sender_missing] = "not_estimable"
    mechanistic_reason.loc[sender_missing] = (
        "mechanistic_sender_assignment_not_estimable"
    )
    assignment_zero = assignment.eq(0.0) & ~parent_zero & ~parent_missing
    mechanistic_status.loc[assignment_zero] = "structural_zero"
    mechanistic_reason.loc[assignment_zero] = "mechanistic_sender_assignment_zero"

    raw_effect = _finite_series(
        downstream_values["differential_effect"], field="differential_effect"
    )
    source_status = downstream_values["_downstream_source_status"].astype("string")
    source_reason = downstream_values["_downstream_source_reason"].astype("string")
    downstream_uninformative = source_reason.isin(
        (
            "zero_receiver_contrast_structural_zero_v1",
            "receptor_family_ineligible",
        )
    )
    downstream_estimable = source_status.eq("observed") & raw_effect.notna()
    downstream_signed = raw_effect.clip(-1.0, 1.0)
    downstream_signed.loc[~downstream_estimable | downstream_uninformative] = np.nan
    downstream_positive = downstream_signed.clip(lower=0.0)
    downstream_status = pd.Series("observed", index=work_index, dtype=object)
    downstream_reason = pd.Series(None, index=work_index, dtype=object)
    downstream_ne = ~downstream_estimable | downstream_uninformative
    downstream_status.loc[downstream_ne] = "not_estimable"
    downstream_reason.loc[downstream_ne] = source_reason.loc[downstream_ne].where(
        source_reason.loc[downstream_ne].notna(),
        "downstream_support_not_estimable",
    )
    downstream_reason.loc[downstream_uninformative] = (
        "receiver_contrast_below_information_floor"
    )

    hybrid = _modulate_vectorized(
        mechanistic_sender,
        downstream_signed,
        modulation_strength=modulation_strength,
    )
    strict_score = _unit_series(
        compact_scores["global_sender_lr_score"], field="global_sender_lr_score"
    )
    strict_status = compact_scores["status"].astype(str)
    strict_reason = compact_scores["reason_code"].copy()

    program_score = _unit_series(
        receiver_program_values["component_value"], field="receiver_program_score"
    )
    program_source_status = receiver_program_values[
        "_receiver_program_source_status"
    ].astype("string")
    program_source_reason = receiver_program_values[
        "_receiver_program_source_reason"
    ].astype("string")
    program_available = (
        program_source_status.isin(("observed", "structural_zero"))
        & program_score.notna()
    )
    program_status = pd.Series("observed", index=work_index, dtype=object)
    program_reason = pd.Series(None, index=work_index, dtype=object)
    program_status.loc[~program_available] = "not_estimable"
    program_reason.loc[~program_available] = program_source_reason.loc[
        ~program_available
    ].where(
        program_source_reason.loc[~program_available].notna(),
        "receiver_program_not_estimable",
    )

    detection_evidence = bounded_detection_evidence_score(
        mechanistic_lr,
        assignment,
        program_score.where(program_available),
    )
    detection_missing = mechanistic_lr.isna() | assignment.isna()
    detection_evidence.loc[detection_missing] = np.nan
    detection_status = pd.Series("observed", index=work_index, dtype=object)
    detection_reason = pd.Series(None, index=work_index, dtype=object)
    detection_status.loc[detection_missing] = "not_estimable"
    detection_reason.loc[detection_missing] = np.where(
        mechanistic_lr.loc[detection_missing].isna(),
        "mechanistic_guard_not_estimable",
        "sender_assignment_not_estimable",
    )
    program_fallback = ~detection_missing & ~program_available
    detection_reason.loc[program_fallback] = (
        "receiver_program_not_estimable_used_sender_only_evidence"
    )
    mechanism_zero = ~detection_missing & mechanistic_lr.eq(0.0)
    detection_reason.loc[mechanism_zero] = (
        "mechanism_structural_zero_soft_penalty_detection_only"
    )

    if resolved_policy in {
        DownstreamEvidencePolicy.DISABLED,
        DownstreamEvidencePolicy.ANNOTATE,
    }:
        selected = mechanistic_sender
        selected_status = mechanistic_status
        selected_reason = mechanistic_reason
    elif resolved_policy is DownstreamEvidencePolicy.MODULATE:
        selected = hybrid
        selected_status = mechanistic_status
        selected_reason = mechanistic_reason
    else:
        selected = strict_score
        selected_status = strict_status
        selected_reason = strict_reason

    result = compact_scores.copy(deep=False)
    result["mechanistic_lr_score"] = mechanistic_lr.to_numpy(copy=False)
    result["mechanistic_assignment_weight"] = assignment.to_numpy(copy=False)
    result["mechanistic_sender_lr_score"] = mechanistic_sender.to_numpy(copy=False)
    result["mechanistic_status"] = mechanistic_status.to_numpy(copy=False)
    result["mechanistic_reason_code"] = mechanistic_reason.to_numpy(copy=False)
    result["downstream_loss_ratio_effect"] = raw_effect.to_numpy(copy=False)
    result["downstream_signed_support"] = downstream_signed.to_numpy(copy=False)
    result["downstream_positive_support"] = downstream_positive.to_numpy(copy=False)
    result["downstream_status"] = downstream_status.to_numpy(copy=False)
    result["downstream_reason_code"] = downstream_reason.to_numpy(copy=False)
    result["downstream_modulated_sender_lr_score"] = hybrid.to_numpy(copy=False)
    result["downstream_confirmed_sender_lr_score"] = strict_score.to_numpy(copy=False)
    result["downstream_confirmed_status"] = strict_status.to_numpy(copy=False)
    result["downstream_confirmed_reason_code"] = strict_reason.to_numpy(copy=False)
    result["receiver_program_score"] = program_score.to_numpy(copy=False)
    result["receiver_program_status"] = program_status.to_numpy(copy=False)
    result["receiver_program_reason_code"] = program_reason.to_numpy(copy=False)
    result["bounded_detection_evidence_score"] = detection_evidence.to_numpy(copy=False)
    result["bounded_detection_evidence_status"] = detection_status.to_numpy(copy=False)
    result["bounded_detection_evidence_reason_code"] = detection_reason.to_numpy(
        copy=False
    )
    result["bounded_detection_evidence_version"] = BOUNDED_DETECTION_EVIDENCE_VERSION
    result["bounded_detection_evidence_formal_inference_allowed"] = False
    result["selected_score"] = selected.to_numpy(copy=False)
    result["selected_score_policy"] = resolved_policy.value
    result["selected_score_status"] = selected_status.to_numpy(copy=False)
    result["selected_score_reason_code"] = selected_reason.to_numpy(copy=False)
    result["score_layer_schema_version"] = SCORE_LAYER_SCHEMA_VERSION

    selected_ne = result["selected_score_status"].astype(str).eq("not_estimable")
    if result.loc[selected_ne, "selected_score"].notna().any():
        raise RuntimeError("not-estimable selected score rows must remain missing")
    return result


def summarize_score_layer(
    table: pd.DataFrame,
    *,
    score_column: str,
    status_column: str,
) -> dict[str, object]:
    """Return explicit coverage and degeneracy diagnostics for one score layer."""

    _require_columns(table, (score_column, status_column), name="score layer")
    score = _finite_series(table[score_column], field=score_column)
    status = table[status_column].astype(str)
    finite = score.dropna()
    finite_rows = len(finite)
    if finite_rows:
        _, counts = np.unique(finite.to_numpy(dtype=float), return_counts=True)
        tied_rows = int(counts[counts > 1].sum())
        unique = len(counts)
    else:
        tied_rows = 0
        unique = 0
    return {
        "rows": len(table),
        "finite_rows": finite_rows,
        "estimable_fraction": finite_rows / len(table),
        "nonzero_rows": int(finite.ne(0.0).sum()),
        "nonzero_fraction": float(finite.ne(0.0).mean()) if finite_rows else 0.0,
        "unique_score_count": unique,
        "tie_fraction": tied_rows / finite_rows if finite_rows else 1.0,
        "degenerate_ranking": unique <= 1,
        "status_counts": {
            str(key): int(value)
            for key, value in status.value_counts(dropna=False).items()
        },
    }


__all__ = [
    "BOUNDED_DETECTION_DOWNSTREAM_WEIGHT",
    "BOUNDED_DETECTION_EVIDENCE_VERSION",
    "BOUNDED_DETECTION_MECHANISM_FLOOR",
    "SCORE_LAYER_SCHEMA_VERSION",
    "SCORE_LAYER_VALUE_COLUMNS",
    "bounded_detection_evidence_score",
    "build_multigroup_score_layers",
    "summarize_score_layer",
]
