"""Training-fold empirical calibration and null-sender attribution for v7."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import canonical_digest

from .contracts_v2 import (
    ParentActivityCalibrationV2,
    SenderAttributionFunctionalV2,
    SenderAttributionV2Spec,
    SenderDetectionCalibrationV2,
    SenderV2CalibrationStatus,
)

_KEY = ("sample_id", "sender", "receiver", "interaction_id")
_PARENT_KEY = ("sample_id", "receiver", "interaction_id")
_CALIBRATION_KEY = ("sender", "receiver", "interaction_id")
_USABLE_STATUSES = {"observed", "low_evidence", "structural_impossible"}
_ALL_STATUSES = {*_USABLE_STATUSES, "not_estimable"}
_REQUIRED_TRAINING_COLUMNS = {
    "sample_id",
    "subject_id",
    "sender",
    "receiver",
    "interaction_id",
    "sender_detection_raw",
    "status",
}
_REQUIRED_APPLICATION_COLUMNS = {
    *_REQUIRED_TRAINING_COLUMNS,
    "context_id",
    "fold_id",
    "active_probability",
    "occurrence_status",
    "occurrence_reason_code",
    "occurrence_functional_id",
    "sender_attribution",
    "null_sender_attribution",
    "attribution_entropy",
    "attribution_status",
    "attribution_reason_code",
    "attribution_functional_id",
}


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    result = tuple(sorted(_name(value, field_name=field_name) for value in values))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return result


def _canonical_table_digest(table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    ordered = table.loc[:, list(columns)].sort_values(list(_KEY), kind="stable")
    records: list[list[Any]] = []
    for row in ordered.itertuples(index=False, name=None):
        record: list[Any] = []
        for value in row:
            if pd.isna(value):
                record.append(None)
            elif isinstance(value, (float, np.floating)):
                record.append({"float_hex": float(value).hex()})
            elif isinstance(value, np.generic):
                record.append(value.item())
            else:
                record.append(value)
        records.append(record)
    return cast(str, canonical_digest(records))


def _validate_common(
    score_table: pd.DataFrame,
    *,
    parent_head: str,
    required: set[str],
) -> pd.DataFrame:
    if not isinstance(score_table, pd.DataFrame):
        raise TypeError("score_table must be a pandas DataFrame")
    missing = required.union({parent_head}).difference(score_table.columns)
    if missing:
        raise ValueError(f"sender v2 input is missing columns: {sorted(missing)}")
    table = score_table.copy(deep=True)
    identifier_columns = [
        "sample_id",
        "subject_id",
        "sender",
        "receiver",
        "interaction_id",
        "status",
    ]
    identifier_columns.extend(
        column for column in ("context_id", "fold_id") if column in table
    )
    for column in identifier_columns:
        if table[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        table[column] = table[column].astype(str)
        if (
            table[column].eq("").any()
            or table[column].str.strip().ne(table[column]).any()
        ):
            raise ValueError(f"{column} must contain canonical identifiers")
    if not set(table["status"]).issubset(_ALL_STATUSES):
        raise ValueError("sender v2 input status is unsupported")
    if table.duplicated(list(_KEY)).any():
        raise ValueError("sender v2 input requires one candidate row per sample")
    sample_subjects = table.groupby("sample_id", observed=True)["subject_id"].nunique()
    if (sample_subjects != 1).any():
        raise ValueError("each sample_id must map to exactly one subject_id")
    for column in ("sender_detection_raw", parent_head):
        numeric = pd.to_numeric(table[column], errors="coerce")
        invalid = table[column].notna() & numeric.isna()
        finite = numeric.dropna()
        if invalid.any() or np.isinf(finite).any() or (finite < 0.0).any():
            raise ValueError(f"{column} must contain non-negative values or NA")
        table[column] = numeric.astype(float)
    usable = table["status"].isin(_USABLE_STATUSES)
    if table.loc[usable, "sender_detection_raw"].isna().any():
        raise ValueError("usable sender rows require sender_detection_raw")
    if table.loc[~usable, "sender_detection_raw"].notna().any():
        raise ValueError("not-estimable sender rows require detection NA")
    parent_variation = table.groupby(list(_PARENT_KEY), observed=True)[
        parent_head
    ].nunique(dropna=False)
    if (parent_variation > 1).any():
        raise ValueError("parent activity must be constant within sample parent")
    return table.reset_index(drop=True)


def _subject_values(
    table: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
) -> pd.Series:
    usable = table[value_column].notna()
    return (
        table.loc[usable]
        .groupby([*group_columns, "subject_id"], observed=True)[value_column]
        .mean()
    )


def fit_sender_attribution_v2(
    training_scores: pd.DataFrame,
    *,
    fold_id: str,
    training_subject_ids: tuple[str, ...],
    training_input_digest: str,
    activity_transform_id: str,
    spec: SenderAttributionV2Spec | None = None,
) -> SenderAttributionFunctionalV2:
    """Fit outcome-agnostic parent ECDF and sender robust scaling on training rows."""

    resolved = spec or SenderAttributionV2Spec()
    if not isinstance(resolved, SenderAttributionV2Spec):
        raise TypeError("spec must be SenderAttributionV2Spec or None")
    fold = _name(fold_id, field_name="fold_id")
    subjects = _names(training_subject_ids, field_name="training_subject_ids")
    training_digest = _name(training_input_digest, field_name="training_input_digest")
    transform_id = _name(activity_transform_id, field_name="activity_transform_id")
    table = _validate_common(
        training_scores,
        parent_head=resolved.parent_activity_head,
        required=_REQUIRED_TRAINING_COLUMNS,
    )
    observed_subjects = tuple(sorted(table["subject_id"].unique()))
    if observed_subjects != subjects:
        raise ValueError("training scores must exactly cover training_subject_ids")
    digest_columns = (
        "sample_id",
        "subject_id",
        "sender",
        "receiver",
        "interaction_id",
        "sender_detection_raw",
        resolved.parent_activity_head,
        "status",
    )
    input_digest = _canonical_table_digest(table, digest_columns)

    parent_sample = table.drop_duplicates(list(_PARENT_KEY))
    parent_values = _subject_values(
        parent_sample,
        value_column=resolved.parent_activity_head,
        group_columns=("receiver", "interaction_id"),
    )
    parent_records: list[ParentActivityCalibrationV2] = []
    parent_keys = sorted(
        {
            (str(row.receiver), str(row.interaction_id))
            for row in parent_sample.itertuples(index=False)
        }
    )
    for receiver, interaction_id in parent_keys:
        try:
            values = parent_values.xs(
                (receiver, interaction_id),
                level=("receiver", "interaction_id"),
            ).to_numpy(dtype=float)
        except KeyError:
            values = np.asarray([], dtype=float)
        n_subjects = len(values)
        observed = n_subjects >= resolved.minimum_calibration_subjects
        parent_records.append(
            ParentActivityCalibrationV2(
                receiver=receiver,
                interaction_id=interaction_id,
                n_subjects=n_subjects,
                sorted_reference_values=(
                    tuple(float(value) for value in np.sort(values)) if observed else ()
                ),
                status=(
                    SenderV2CalibrationStatus.OBSERVED
                    if observed
                    else SenderV2CalibrationStatus.NOT_ESTIMABLE
                ),
                reason_code=(
                    None if observed else "insufficient_parent_calibration_subjects"
                ),
            )
        )

    detection_values = _subject_values(
        table,
        value_column="sender_detection_raw",
        group_columns=_CALIBRATION_KEY,
    )
    detection_records: list[SenderDetectionCalibrationV2] = []
    detection_keys = sorted(
        {
            (str(row.sender), str(row.receiver), str(row.interaction_id))
            for row in table.itertuples(index=False)
        }
    )
    for sender, receiver, interaction_id in detection_keys:
        try:
            values = detection_values.xs(
                (sender, receiver, interaction_id),
                level=_CALIBRATION_KEY,
            ).to_numpy(dtype=float)
        except KeyError:
            values = np.asarray([], dtype=float)
        n_subjects = len(values)
        observed = n_subjects >= resolved.minimum_calibration_subjects
        center: float | None
        scale: float | None
        if observed:
            pooled_values = detection_values.xs(
                (receiver, interaction_id),
                level=("receiver", "interaction_id"),
            ).to_numpy(dtype=float)
            center = float(np.median(pooled_values))
            mad = float(np.median(np.abs(pooled_values - center)))
            scale = max(
                resolved.minimum_detection_scale,
                resolved.mad_scale * mad,
            )
        else:
            center = scale = None
        detection_records.append(
            SenderDetectionCalibrationV2(
                sender=sender,
                receiver=receiver,
                interaction_id=interaction_id,
                n_subjects=n_subjects,
                center=center,
                scale=scale,
                status=(
                    SenderV2CalibrationStatus.OBSERVED
                    if observed
                    else SenderV2CalibrationStatus.NOT_ESTIMABLE
                ),
                reason_code=(
                    None if observed else "insufficient_sender_calibration_subjects"
                ),
            )
        )
    return SenderAttributionFunctionalV2(
        fold_id=fold,
        training_subject_ids=subjects,
        training_input_digest=training_digest,
        activity_transform_id=transform_id,
        input_digest=input_digest,
        parent_calibrations=tuple(parent_records),
        detection_calibrations=tuple(detection_records),
        spec=resolved,
    )


def _entmax15(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 1 or len(logits) == 0 or np.any(~np.isfinite(logits)):
        raise ValueError("entmax logits must be a finite non-empty vector")
    shifted = 0.5 * (logits - float(logits.max()))
    lower = float(shifted.min() - 1.0)
    upper = float(shifted.max())
    for _ in range(80):
        threshold = 0.5 * (lower + upper)
        mass = float(np.square(np.maximum(shifted - threshold, 0.0)).sum())
        if mass > 1.0:
            lower = threshold
        else:
            upper = threshold
    weights = np.square(np.maximum(shifted - upper, 0.0))
    total = float(weights.sum())
    if total <= 0.0:
        raise FloatingPointError("entmax normalization produced zero mass")
    return cast(np.ndarray, weights / total)


def _empirical_active_probability(
    value: float,
    calibration: ParentActivityCalibrationV2,
    *,
    pseudocount: float,
) -> float:
    reference = np.asarray(calibration.sorted_reference_values, dtype=float)
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    midrank = 0.5 * (left + right)
    return float((midrank + pseudocount) / (len(reference) + 2.0 * pseudocount))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    if len(probabilities) <= 1:
        return 0.0
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(probabilities))
    return min(1.0, max(0.0, entropy))


def apply_sender_attribution_v2(
    functional: SenderAttributionFunctionalV2,
    score_table: pd.DataFrame,
    *,
    activity_transform_id: str,
) -> pd.DataFrame:
    """Apply null-sender 1.5-entmax attribution without altering raw detection."""

    if not isinstance(functional, SenderAttributionFunctionalV2):
        raise TypeError("functional must be SenderAttributionFunctionalV2")
    transform_id = _name(activity_transform_id, field_name="activity_transform_id")
    if transform_id != functional.activity_transform_id:
        raise ValueError("application activity transform does not match functional")
    table = _validate_common(
        score_table,
        parent_head=functional.spec.parent_activity_head,
        required=_REQUIRED_APPLICATION_COLUMNS,
    )
    if set(table["fold_id"].astype(str)) != {functional.fold_id}:
        raise ValueError("application fold_id does not match sender functional")
    for status_column in ("occurrence_status", "attribution_status"):
        if not table[status_column].eq("not_computed").all():
            raise ValueError(f"sender v2 refuses to overwrite {status_column}")
    parent_by_key = {
        (item.receiver, item.interaction_id): item
        for item in functional.parent_calibrations
    }
    detection_by_key = {
        (item.sender, item.receiver, item.interaction_id): item
        for item in functional.detection_calibrations
    }
    output = table.copy(deep=True)
    for _, indices in output.groupby(
        ["sample_id", "context_id", "fold_id", "receiver", "interaction_id"],
        observed=True,
        sort=False,
    ).groups.items():
        group_index = list(indices)
        first = output.loc[group_index[0]]
        parent_key = (str(first["receiver"]), str(first["interaction_id"]))
        parent_calibration = parent_by_key.get(parent_key)
        parent_value = first[functional.spec.parent_activity_head]
        if (
            parent_calibration is None
            or parent_calibration.status is not SenderV2CalibrationStatus.OBSERVED
            or pd.isna(parent_value)
        ):
            reason = (
                "parent_calibration_missing"
                if parent_calibration is None
                else parent_calibration.reason_code
                if parent_calibration.status is SenderV2CalibrationStatus.NOT_ESTIMABLE
                else "parent_activity_not_estimable"
            )
            output.loc[group_index, "occurrence_status"] = "not_estimable"
            output.loc[group_index, "occurrence_reason_code"] = reason
            output.loc[group_index, "occurrence_functional_id"] = (
                functional.functional_id
            )
            output.loc[group_index, "attribution_status"] = "not_estimable"
            output.loc[group_index, "attribution_reason_code"] = reason
            output.loc[group_index, "attribution_functional_id"] = (
                functional.functional_id
            )
            continue
        active_probability = _empirical_active_probability(
            float(parent_value),
            parent_calibration,
            pseudocount=functional.spec.ecdf_pseudocount,
        )
        output.loc[group_index, "active_probability"] = active_probability
        output.loc[group_index, "occurrence_status"] = "partial"
        output.loc[group_index, "occurrence_reason_code"] = None
        output.loc[group_index, "occurrence_functional_id"] = functional.functional_id

        valid_indices: list[int] = []
        logits: list[float] = []
        missing_reasons: dict[int, str] = {}
        for index in group_index:
            row = output.loc[index]
            detection_key = (
                str(row["sender"]),
                str(row["receiver"]),
                str(row["interaction_id"]),
            )
            calibration = detection_by_key.get(detection_key)
            if pd.isna(row["sender_detection_raw"]):
                missing_reasons[index] = "sender_detection_not_estimable"
            elif calibration is None:
                missing_reasons[index] = "sender_calibration_missing"
            elif calibration.status is SenderV2CalibrationStatus.NOT_ESTIMABLE:
                missing_reasons[index] = cast(str, calibration.reason_code)
            else:
                assert calibration.center is not None
                assert calibration.scale is not None
                standardized = (
                    (float(row["sender_detection_raw"]) - calibration.center)
                    / calibration.scale
                    / functional.spec.attribution_temperature
                )
                logits.append(
                    float(
                        np.clip(
                            standardized,
                            -functional.spec.maximum_abs_logit,
                            functional.spec.maximum_abs_logit,
                        )
                    )
                )
                valid_indices.append(index)
        output.loc[group_index, "attribution_functional_id"] = functional.functional_id
        if not valid_indices:
            output.loc[group_index, "attribution_status"] = "not_estimable"
            output.loc[group_index, "attribution_reason_code"] = (
                "no_estimable_sender_calibrations"
            )
            continue
        conditional_weights = _entmax15(np.asarray(logits, dtype=float))
        sender_weights = active_probability * conditional_weights
        null_weight = 1.0 - active_probability
        output.loc[group_index, "null_sender_attribution"] = null_weight
        probabilities = np.concatenate(([null_weight], sender_weights))
        output.loc[group_index, "attribution_entropy"] = _normalized_entropy(
            probabilities
        )
        partial = bool(missing_reasons)
        for index, weight in zip(valid_indices, sender_weights, strict=True):
            output.loc[index, "sender_attribution"] = float(weight)
            output.loc[index, "attribution_status"] = (
                "partial" if partial else "observed"
            )
            output.loc[index, "attribution_reason_code"] = None
        for index, reason in missing_reasons.items():
            output.loc[index, "sender_attribution"] = np.nan
            output.loc[index, "attribution_status"] = "not_estimable"
            output.loc[index, "attribution_reason_code"] = reason
    return output.loc[:, score_table.columns].copy(deep=True)


__all__ = [
    "apply_sender_attribution_v2",
    "fit_sender_attribution_v2",
]
