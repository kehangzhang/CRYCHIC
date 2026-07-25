"""Auditable diagnostics for the v7 sample-level estimator.

The diagnostics are descriptive benchmark artifacts.  They never promote an
analytic p-value to formal inference and never reinterpret missing coverage as
a structural zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from crychic.core import canonical_digest, canonical_json, stable_id
from crychic.inference import (
    CANDIDATE_SENDER_BIAS_COLUMNS,
    RESOLUTION_PERFORMANCE_COLUMNS,
    SCORE_GEOMETRY_COLUMNS,
    V7_DIAGNOSTIC_SCORE_HEADS,
    summarize_v7_candidate_sender_bias,
    summarize_v7_resolution_performance,
    summarize_v7_score_geometry,
)

from .crossfit import CrossFitArtifacts

V7_DIAGNOSTICS_VERSION = "v7_sample_level_diagnostics_v1"
_SCHEMA_VERSION = "1.0.0"

GATE_STAGES = (
    "common_candidate_universe",
    "measurable_ligand_receptor",
    "receptor_eligible",
    "ligand_contrast_supported",
    "family_selected",
    "downstream_supported",
    "sender_assignment_available",
    "native_nonzero",
    "significant",
)
_LEGACY_GATE_STAGES = GATE_STAGES[2:-1]
_STAGE_STATUSES = {"passed", "failed", "not_estimable", "not_computed"}
_STAGE_PRECEDENCE = {
    "not_computed": 0,
    "not_estimable": 1,
    "failed": 2,
    "passed": 3,
}
_STAGE_STATUS_CODE = {
    "not_computed": np.int8(0),
    "not_estimable": np.int8(1),
    "failed": np.int8(2),
    "passed": np.int8(3),
}
_NOT_COMPUTED_CODE = _STAGE_STATUS_CODE["not_computed"]
_NOT_ESTIMABLE_CODE = _STAGE_STATUS_CODE["not_estimable"]
_FAILED_CODE = _STAGE_STATUS_CODE["failed"]
_PASSED_CODE = _STAGE_STATUS_CODE["passed"]
_STATUS_BY_CODE = {int(code): status for status, code in _STAGE_STATUS_CODE.items()}

_DEFAULT_SCORE_HEADS = V7_DIAGNOSTIC_SCORE_HEADS
_ALLOWED_SCORE_HEADS = set(_DEFAULT_SCORE_HEADS)

GATE_ATTRITION_COLUMNS = (
    "dataset_id",
    "contrast_name",
    "fold_id",
    "stratum_kind",
    "stratum_value",
    "stage_order",
    "stage",
    "n_universe",
    "n_stage_evaluable",
    "n_stage_passed",
    "n_stage_failed",
    "n_not_estimable",
    "n_not_computed",
    "n_cumulative_retained",
    "stage_evaluable_fraction",
    "stage_pass_fraction",
    "retention_fraction_from_previous",
    "retention_fraction_from_universe",
)
_GATE_STAGE_LEDGER_STATUS_COLUMNS = tuple(
    field
    for stage in GATE_STAGES
    for field in (f"{stage}_status", f"{stage}_reason_code")
)
GATE_STAGE_LEDGER_COLUMNS = (
    "dataset_id",
    "contrast_name",
    "fold_id",
    "sample_id",
    "subject_id",
    "condition",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    *_GATE_STAGE_LEDGER_STATUS_COLUMNS,
)

_BASE_KEY = (
    "fold_id",
    "sample_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_STAGE_KEY = ("contrast_name", *_BASE_KEY)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if not is_scalar(value):
        return False
    return bool(pd.isna(cast(Any, value)))


def _finite_or_none(value: object) -> float | None:
    if _is_missing_scalar(value):
        return None
    numeric = float(cast(Any, value))
    return numeric if math.isfinite(numeric) else None


def _canonical_frame_digest(table: pd.DataFrame) -> str:
    columns = tuple(sorted(map(str, table.columns)))
    if len(columns) != len(set(columns)):
        raise ValueError("diagnostic tables require unique string column names")
    rows: list[dict[str, object]] = []
    for values in table.loc[:, list(columns)].itertuples(index=False, name=None):
        row: dict[str, object] = {}
        for column, value in zip(columns, values, strict=True):
            if _is_missing_scalar(value):
                normalized: object = None
            elif isinstance(value, float | np.floating):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("diagnostic tables cannot contain infinity")
                normalized = {"float_hex": numeric.hex()}
            elif isinstance(value, bool | np.bool_):
                normalized = bool(value)
            elif isinstance(value, int | np.integer):
                normalized = int(value)
            else:
                normalized = str(value) if is_scalar(value) else canonical_json(value)
            row[str(column)] = normalized
        rows.append(row)
    rows.sort(key=canonical_json)
    return str(canonical_digest({"columns": list(columns), "records": rows}))


def _candidate_sender_bin(value: object) -> str:
    numeric = _finite_or_none(value)
    if numeric is None:
        return "unknown"
    count = int(numeric)
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return ">10"


def _cell_count_bin(value: object) -> str:
    numeric = _finite_or_none(value)
    if numeric is None:
        return "unknown"
    count = int(numeric)
    if count < 20:
        return "0-19"
    if count < 50:
        return "20-49"
    if count < 100:
        return "50-99"
    if count < 500:
        return "100-499"
    return ">=500"


def _mechanism_class(value: object) -> str:
    if _is_missing_scalar(value):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if "ecm" in text or "matrix" in text:
        return "ecm"
    if "contact" in text or "cell-cell" in text or "cell cell" in text:
        return "contact"
    if "secret" in text or "diffus" in text or "cytokine" in text:
        return "secreted"
    return text


def _truth_class(value: object) -> str:
    if _is_missing_scalar(value):
        return "unknown"
    return "truth_positive" if bool(value) else "truth_negative"


def _validate_boolean_series(values: pd.Series, *, field_name: str) -> None:
    if values.isna().any() or any(
        not isinstance(value, bool | np.bool_) for value in values.tolist()
    ):
        raise ValueError(f"{field_name} must contain non-missing booleans")


def _validate_truth_table(
    truth: pd.DataFrame | None,
    *,
    contrast_names: tuple[str, ...],
) -> pd.DataFrame:
    columns = (
        "contrast_name",
        "sender",
        "receiver",
        "interaction_id",
        "truth",
        "truth_effect",
        "mechanism_class",
        "pathway",
    )
    if truth is None:
        return pd.DataFrame(columns=columns)
    if not isinstance(truth, pd.DataFrame):
        raise TypeError("truth must be a pandas DataFrame or None")
    required = {"sender", "receiver", "interaction_id", "truth"}
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"truth is missing columns: {sorted(missing)}")
    source = truth.copy(deep=True)
    if "contrast_name" not in source:
        if len(contrast_names) != 1:
            raise ValueError(
                "multi-contrast truth requires an explicit contrast_name column"
            )
        source["contrast_name"] = contrast_names[0]
    if not set(source["contrast_name"].astype(str)).issubset(contrast_names):
        raise ValueError("truth contains an undeclared contrast_name")
    for column in ("contrast_name", "sender", "receiver", "interaction_id"):
        if source[column].isna().any():
            raise ValueError(f"truth.{column} cannot contain missing values")
        source[column] = source[column].astype(str)
    _validate_boolean_series(source["truth"], field_name="truth.truth")
    key = ["contrast_name", "sender", "receiver", "interaction_id"]
    if source.duplicated(key).any():
        raise ValueError("truth child keys must be unique")
    if "truth_effect" not in source:
        source["truth_effect"] = np.nan
    else:
        numeric = pd.to_numeric(source["truth_effect"], errors="coerce")
        invalid = source["truth_effect"].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.dropna()).any():
            raise ValueError("truth_effect must be finite or missing")
        source["truth_effect"] = numeric
    for column in ("mechanism_class", "pathway"):
        if column not in source:
            source[column] = None
    return source.loc[:, list(columns)]


def _validate_cell_counts(cell_counts: pd.DataFrame | None) -> pd.DataFrame:
    columns = ("sample_id", "cell_type", "cell_count")
    if cell_counts is None:
        return pd.DataFrame(columns=columns)
    if not isinstance(cell_counts, pd.DataFrame):
        raise TypeError("cell_counts must be a pandas DataFrame or None")
    missing = set(columns).difference(cell_counts.columns)
    if missing:
        raise ValueError(f"cell_counts is missing columns: {sorted(missing)}")
    source = cell_counts.loc[:, list(columns)].copy()
    for column in ("sample_id", "cell_type"):
        if source[column].isna().any():
            raise ValueError(f"cell_counts.{column} cannot contain missing values")
        source[column] = source[column].astype(str)
    numeric = pd.to_numeric(source["cell_count"], errors="coerce")
    if numeric.isna().any() or np.isinf(numeric).any() or numeric.lt(0).any():
        raise ValueError("cell_count must contain finite non-negative values")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("cell_count must contain integer values")
    source["cell_count"] = numeric.astype(int)
    if source.duplicated(["sample_id", "cell_type"]).any():
        raise ValueError("cell_counts sample/cell-type keys must be unique")
    return source


def _interaction_annotations(crossfit: CrossFitArtifacts) -> pd.DataFrame:
    records: dict[str, dict[str, object]] = {}
    for fold in crossfit.folds:
        for interaction in fold.training.resource_bundle.interactions:
            record = {
                "interaction_id": str(interaction.interaction_id),
                "pathway_resource": interaction.pathway,
                "mechanism_resource": _mechanism_class(interaction.annotation),
                "receptor_complex_cardinality": len(interaction.receptor_subunits),
            }
            previous = records.get(str(interaction.interaction_id))
            if previous is not None and previous != record:
                raise ValueError("interaction annotations differ across outer folds")
            records[str(interaction.interaction_id)] = record
    return pd.DataFrame.from_records(
        list(records.values()),
        columns=(
            "interaction_id",
            "pathway_resource",
            "mechanism_resource",
            "receptor_complex_cardinality",
        ),
    )


def _decorate_scores(
    crossfit: CrossFitArtifacts,
    *,
    truth: pd.DataFrame | None,
    cell_counts: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crossfit._require_intact()
    scores = crossfit.oof_sample_edge_scores_v2
    if scores.empty:
        raise ValueError("v7 diagnostics require M0 v2 scores in every fold")
    if scores.duplicated(list(_BASE_KEY)).any():
        raise ValueError("OOF v7 score child keys must be unique")
    contrasts = tuple(sorted(item.name for item in crossfit.spec.contrasts))
    if not contrasts:
        raise ValueError("v7 diagnostics require at least one declared contrast")
    truth_table = _validate_truth_table(truth, contrast_names=contrasts)
    count_table = _validate_cell_counts(cell_counts)

    expanded = pd.concat(
        [scores.assign(contrast_name=name) for name in contrasts],
        ignore_index=True,
    )
    expanded = expanded.merge(
        _interaction_annotations(crossfit),
        on="interaction_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if truth_table.empty:
        expanded["truth"] = pd.NA
        expanded["truth_effect"] = np.nan
        expanded["mechanism_class_truth"] = None
        expanded["pathway_truth"] = None
    else:
        renamed_truth = truth_table.rename(
            columns={
                "mechanism_class": "mechanism_class_truth",
                "pathway": "pathway_truth",
            }
        )
        expanded = expanded.merge(
            renamed_truth,
            on=["contrast_name", "sender", "receiver", "interaction_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
    expanded["truth_class"] = expanded["truth"].map(_truth_class)
    mechanism = expanded["mechanism_class_truth"].where(
        expanded["mechanism_class_truth"].notna(),
        expanded["mechanism_resource"],
    )
    expanded["mechanism_class"] = mechanism.map(_mechanism_class)
    expanded["pathway"] = expanded["pathway_truth"].where(
        expanded["pathway_truth"].notna(),
        expanded["pathway_resource"],
    )

    if count_table.empty:
        expanded["sender_cell_count"] = np.nan
        expanded["receiver_cell_count"] = np.nan
    else:
        sender_counts = count_table.rename(
            columns={"cell_type": "sender", "cell_count": "sender_cell_count"}
        )
        receiver_counts = count_table.rename(
            columns={
                "cell_type": "receiver",
                "cell_count": "receiver_cell_count",
            }
        )
        expanded = expanded.merge(
            sender_counts,
            on=["sample_id", "sender"],
            how="left",
            validate="many_to_one",
            sort=False,
        ).merge(
            receiver_counts,
            on=["sample_id", "receiver"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
    expanded["minimum_cell_count"] = expanded[
        ["sender_cell_count", "receiver_cell_count"]
    ].min(axis=1, skipna=False)
    expanded["cell_count_bin"] = expanded["minimum_cell_count"].map(_cell_count_bin)
    expanded["candidate_sender_bin"] = expanded["candidate_sender_count"].map(
        _candidate_sender_bin
    )
    if expanded["receptor_complex_cardinality"].isna().any():
        raise ValueError("OOF scores contain an interaction absent from the resource")
    expanded["receptor_complex_cardinality"] = expanded[
        "receptor_complex_cardinality"
    ].astype(int)
    expanded = expanded.sort_values(
        ["contrast_name", *_BASE_KEY], kind="stable", ignore_index=True
    )
    return expanded, truth_table, count_table


def _first_reason(values: pd.Series) -> str | None:
    present = values.dropna().astype(str)
    present = present.loc[present.ne("")]
    return None if present.empty else str(sorted(set(present))[0])


def _combine_stage_record(
    target: dict[tuple[str, ...], tuple[str, str | None]],
    key: tuple[str, ...],
    status: str,
    reason: str | None,
) -> None:
    if status not in _STAGE_STATUSES:
        raise ValueError(f"unsupported gate stage status: {status}")
    previous = target.get(key)
    if previous is None or _STAGE_PRECEDENCE[status] > _STAGE_PRECEDENCE[previous[0]]:
        target[key] = (status, reason)


def _boolean_stage_status(values: pd.Series) -> str:
    observed = values.dropna()
    if observed.empty:
        return "not_estimable"
    return "passed" if observed.astype(bool).any() else "failed"


def _legacy_gate_records(
    crossfit: CrossFitArtifacts,
) -> dict[tuple[str, ...], tuple[str, str | None]]:
    result: dict[tuple[str, ...], tuple[str, str | None]] = {}
    child_group_columns = [
        "sample_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    ]
    parent_group_columns = [
        "sample_id",
        "context_id",
        "receiver",
        "interaction_id",
    ]
    family_join = [
        "family_common_functional_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "mode",
    ]
    for fold in crossfit.folds:
        if fold.sample_edge_scores_v2 is None:
            raise ValueError("legacy gate diagnostics require fold M0 v2 candidates")
        candidate_senders = (
            fold.sample_edge_scores_v2.table.loc[:, [*parent_group_columns, "sender"]]
            .drop_duplicates()
            .groupby(parent_group_columns, observed=True)["sender"]
            .agg(lambda values: tuple(sorted(values.astype(str))))
            .to_dict()
        )
        for application in fold.family_common_applications:
            contrast = str(application.functional.contrast_name)
            members = application.member_scores
            families = application.family_scores
            senders = application.sender_scores
            if "mode" in members:
                members = members.loc[members["mode"].astype(str).eq("state")]
            if "mode" in families:
                families = families.loc[families["mode"].astype(str).eq("state")]
            if "mode" in senders:
                senders = senders.loc[senders["mode"].astype(str).eq("state")]

            merged = members.merge(
                families.loc[
                    :,
                    [
                        *family_join,
                        "family_selected",
                        "incremental_downstream_gain",
                        "status",
                        "reason_code",
                    ],
                ].rename(
                    columns={
                        "status": "family_status",
                        "reason_code": "family_reason_code",
                    }
                ),
                on=family_join,
                how="left",
                validate="many_to_one",
                sort=False,
            )
            for values, group in merged.groupby(
                parent_group_columns, observed=True, sort=False
            ):
                receptor_status = _boolean_stage_status(group["receptor_eligible"])
                ligand_statuses = set(
                    group["ligand_contrast_gate_status"].dropna().astype(str)
                )
                if "supported" in ligand_statuses:
                    ligand_status = "passed"
                elif "unsupported" in ligand_statuses:
                    ligand_status = "failed"
                else:
                    ligand_status = "not_estimable"
                family_status = _boolean_stage_status(group["family_selected"])
                gain = pd.to_numeric(
                    group["incremental_downstream_gain"], errors="coerce"
                )
                observed_gain = gain.dropna()
                if observed_gain.empty:
                    downstream_status = "not_estimable"
                else:
                    downstream_status = (
                        "passed" if observed_gain.gt(0.0).any() else "failed"
                    )
                parent_key = tuple(str(value) for value in values)
                for sender in candidate_senders.get(parent_key, ()):
                    sample_id, context_id, receiver, interaction_id = parent_key
                    parent_child_key = (
                        contrast,
                        str(fold.fold_id),
                        sample_id,
                        context_id,
                        sender,
                        receiver,
                        interaction_id,
                    )
                    _combine_stage_record(
                        result,
                        (*parent_child_key, "receptor_eligible"),
                        receptor_status,
                        _first_reason(group["reason_code"]),
                    )
                    _combine_stage_record(
                        result,
                        (*parent_child_key, "ligand_contrast_supported"),
                        ligand_status,
                        _first_reason(group["ligand_contrast_gate_reason_code"]),
                    )
                    _combine_stage_record(
                        result,
                        (*parent_child_key, "family_selected"),
                        family_status,
                        _first_reason(group["family_reason_code"]),
                    )
                    _combine_stage_record(
                        result,
                        (*parent_child_key, "downstream_supported"),
                        downstream_status,
                        _first_reason(group["family_reason_code"]),
                    )

            for values, group in senders.groupby(
                child_group_columns, observed=True, sort=False
            ):
                sender_child_key = (
                    contrast,
                    str(fold.fold_id),
                    *(str(value) for value in values),
                )
                assignment = pd.to_numeric(group["assignment_weight"], errors="coerce")
                resolved = pd.to_numeric(
                    group["sender_resolved_strength"], errors="coerce"
                )
                assignment_status = (
                    "passed" if assignment.notna().any() else "not_estimable"
                )
                native_status = (
                    "not_estimable"
                    if resolved.notna().sum() == 0
                    else ("passed" if resolved.gt(0.0).any() else "failed")
                )
                reason = _first_reason(group["reason_code"])
                _combine_stage_record(
                    result,
                    (*sender_child_key, "sender_assignment_available"),
                    assignment_status,
                    reason,
                )
                _combine_stage_record(
                    result,
                    (*sender_child_key, "native_nonzero"),
                    native_status,
                    reason,
                )
    return result


def _validate_gate_annotations(
    annotations: pd.DataFrame | None,
    *,
    contrast_names: set[str],
) -> pd.DataFrame:
    columns = (*_STAGE_KEY, "stage", "stage_status", "reason_code")
    if annotations is None:
        return pd.DataFrame(columns=columns)
    if not isinstance(annotations, pd.DataFrame):
        raise TypeError("gate_annotations must be a pandas DataFrame or None")
    required = set(columns).difference({"reason_code"})
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"gate_annotations is missing columns: {sorted(missing)}")
    source = annotations.copy(deep=True)
    if "reason_code" not in source:
        source["reason_code"] = None
    for column in _STAGE_KEY:
        if source[column].isna().any():
            raise ValueError(f"gate_annotations.{column} cannot be missing")
        source[column] = source[column].astype(str)
    if not set(source["contrast_name"]).issubset(contrast_names):
        raise ValueError("gate_annotations contains an undeclared contrast")
    if not set(source["stage"]).issubset(_LEGACY_GATE_STAGES):
        raise ValueError(
            "gate_annotations may override only legacy intermediate stages"
        )
    if not set(source["stage_status"]).issubset(_STAGE_STATUSES):
        raise ValueError("gate_annotations contains an unsupported stage_status")
    if source.duplicated([*_STAGE_KEY, "stage"]).any():
        raise ValueError("gate_annotations keys must be unique")
    return source.loc[:, list(columns)]


def _validate_significance(
    significance: pd.DataFrame | None,
    *,
    contrast_names: set[str],
) -> pd.DataFrame:
    columns = (*_STAGE_KEY, "significant", "reason_code")
    if significance is None:
        return pd.DataFrame(columns=columns)
    if not isinstance(significance, pd.DataFrame):
        raise TypeError("significance must be a pandas DataFrame or None")
    required = set(columns).difference({"reason_code"})
    missing = required.difference(significance.columns)
    if missing:
        raise ValueError(f"significance is missing columns: {sorted(missing)}")
    source = significance.copy(deep=True)
    if "reason_code" not in source:
        source["reason_code"] = None
    for column in _STAGE_KEY:
        if source[column].isna().any():
            raise ValueError(f"significance.{column} cannot be missing")
        source[column] = source[column].astype(str)
    if not set(source["contrast_name"]).issubset(contrast_names):
        raise ValueError("significance contains an undeclared contrast")
    _validate_boolean_series(
        source["significant"], field_name="significance.significant"
    )
    if source.duplicated(list(_STAGE_KEY)).any():
        raise ValueError("significance child keys must be unique")
    return source.loc[:, list(columns)]


def _stage_record_map(
    crossfit: CrossFitArtifacts,
    *,
    gate_annotations: pd.DataFrame | None,
    significance: pd.DataFrame | None,
) -> dict[tuple[str, ...], tuple[str, str | None]]:
    contrasts = {item.name for item in crossfit.spec.contrasts}
    result = _legacy_gate_records(crossfit)
    annotations = _validate_gate_annotations(gate_annotations, contrast_names=contrasts)
    for row in annotations.itertuples(index=False):
        key = tuple(str(getattr(row, column)) for column in _STAGE_KEY)
        result[(*key, str(row.stage))] = (
            str(row.stage_status),
            None if pd.isna(row.reason_code) else str(row.reason_code),
        )
    significance_table = _validate_significance(significance, contrast_names=contrasts)
    for row in significance_table.itertuples(index=False):
        key = tuple(str(getattr(row, column)) for column in _STAGE_KEY)
        result[(*key, "significant")] = (
            "passed" if bool(row.significant) else "failed",
            None if pd.isna(row.reason_code) else str(row.reason_code),
        )
    return result


def _stage_arrays(
    scores: pd.DataFrame,
    stage_records: Mapping[tuple[str, ...], tuple[str, str | None]],
) -> dict[str, np.ndarray]:
    statuses: dict[str, np.ndarray] = {}
    statuses["common_candidate_universe"] = np.full(
        len(scores), _PASSED_CODE, dtype=np.int8
    )
    measurable = (
        scores["ligand_activity_raw"].notna() & scores["receptor_activity_raw"].notna()
    )
    statuses["measurable_ligand_receptor"] = np.where(
        measurable, _PASSED_CODE, _NOT_ESTIMABLE_CODE
    ).astype(np.int8)
    keys = [
        tuple(str(value) for value in row)
        for row in scores.loc[:, list(_STAGE_KEY)].itertuples(index=False, name=None)
    ]
    for stage in (*_LEGACY_GATE_STAGES, "significant"):
        stage_status: np.ndarray = np.empty(len(scores), dtype=np.int8)
        for index, key in enumerate(keys):
            status, _ = stage_records.get(
                (*key, stage), ("not_computed", f"{stage}_not_computed")
            )
            stage_status[index] = _STAGE_STATUS_CODE[status]
        statuses[stage] = stage_status
    return statuses


def _gate_stage_ledger(
    scores: pd.DataFrame,
    *,
    dataset_id: str,
    statuses: Mapping[str, np.ndarray],
    stage_records: Mapping[tuple[str, ...], tuple[str, str | None]],
) -> pd.DataFrame:
    base_columns = (
        "contrast_name",
        "fold_id",
        "sample_id",
        "subject_id",
        "condition",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
    )
    result = scores.loc[:, list(base_columns)].copy(deep=True)
    result.insert(0, "dataset_id", dataset_id)
    keys = [
        tuple(str(value) for value in row)
        for row in scores.loc[:, list(_STAGE_KEY)].itertuples(index=False, name=None)
    ]
    for stage in GATE_STAGES:
        stage_values = statuses[stage]
        result[f"{stage}_status"] = [
            _STATUS_BY_CODE[int(value)] for value in stage_values
        ]
        if stage == "common_candidate_universe":
            reasons: list[str | None] = [None] * len(result)
        elif stage == "measurable_ligand_receptor":
            reasons = [
                None
                if int(value) == int(_PASSED_CODE)
                else "ligand_or_receptor_not_measurable"
                for value in stage_values
            ]
        else:
            reasons = [
                stage_records.get(
                    (*key, stage),
                    ("not_computed", f"{stage}_not_computed"),
                )[1]
                for key in keys
            ]
        result[f"{stage}_reason_code"] = reasons
    result = result.loc[:, list(GATE_STAGE_LEDGER_COLUMNS)].sort_values(
        [
            "dataset_id",
            "contrast_name",
            "fold_id",
            "sample_id",
            "sender",
            "receiver",
            "interaction_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    if result.duplicated(
        [
            "contrast_name",
            "fold_id",
            "sample_id",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
        ]
    ).any():
        raise ValueError("gate-stage ledger child keys must be unique")
    return result


def build_v7_gate_stage_ledger(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    gate_annotations: pd.DataFrame | None = None,
    significance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Materialize exact per-child legacy gate states for component swaps."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    dataset = _name(dataset_id, field_name="dataset_id")
    scores, _, _ = _decorate_scores(crossfit, truth=None, cell_counts=None)
    stage_records = _stage_record_map(
        crossfit,
        gate_annotations=gate_annotations,
        significance=significance,
    )
    return _gate_stage_ledger(
        scores,
        dataset_id=dataset,
        statuses=_stage_arrays(scores, stage_records),
        stage_records=stage_records,
    )


def _strata_masks(scope: pd.DataFrame) -> Iterable[tuple[str, str, np.ndarray]]:
    yield "overall", "all", np.ones(len(scope), dtype=bool)
    columns = (
        "truth_class",
        "mechanism_class",
        "candidate_sender_bin",
        "cell_count_bin",
        "receptor_complex_cardinality",
    )
    for column in columns:
        values = scope[column].astype(str).to_numpy()
        for value in sorted(set(values)):
            yield column, value, values == value


def _gate_attrition(
    scores: pd.DataFrame,
    *,
    dataset_id: str,
    statuses: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    fold_scopes = [*sorted(scores["fold_id"].astype(str).unique()), "__all__"]
    for contrast in sorted(scores["contrast_name"].astype(str).unique()):
        contrast_mask = scores["contrast_name"].astype(str).eq(contrast).to_numpy()
        for fold_id in fold_scopes:
            fold_mask = (
                np.ones(len(scores), dtype=bool)
                if fold_id == "__all__"
                else scores["fold_id"].astype(str).eq(fold_id).to_numpy()
            )
            scope_positions = np.flatnonzero(contrast_mask & fold_mask)
            scope = scores.iloc[scope_positions].reset_index(drop=True)
            if scope.empty:
                continue
            scoped_statuses = {
                stage: values[scope_positions] for stage, values in statuses.items()
            }
            cumulative: np.ndarray = np.ones(len(scope), dtype=bool)
            previous: np.ndarray = np.ones(len(scope), dtype=bool)
            for stage_order, stage in enumerate(GATE_STAGES, start=1):
                stage_values = scoped_statuses[stage]
                cumulative = cumulative & (stage_values == _PASSED_CODE)
                for stratum_kind, stratum_value, stratum_mask in _strata_masks(scope):
                    n_universe = int(np.count_nonzero(stratum_mask))
                    selected = stage_values[stratum_mask]
                    n_passed = int(np.count_nonzero(selected == _PASSED_CODE))
                    n_failed = int(np.count_nonzero(selected == _FAILED_CODE))
                    n_ne = int(np.count_nonzero(selected == _NOT_ESTIMABLE_CODE))
                    n_nc = int(np.count_nonzero(selected == _NOT_COMPUTED_CODE))
                    n_evaluable = n_passed + n_failed
                    n_retained = int(np.count_nonzero(cumulative[stratum_mask]))
                    n_previous = int(np.count_nonzero(previous[stratum_mask]))
                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "contrast_name": contrast,
                            "fold_id": fold_id,
                            "stratum_kind": stratum_kind,
                            "stratum_value": stratum_value,
                            "stage_order": stage_order,
                            "stage": stage,
                            "n_universe": n_universe,
                            "n_stage_evaluable": n_evaluable,
                            "n_stage_passed": n_passed,
                            "n_stage_failed": n_failed,
                            "n_not_estimable": n_ne,
                            "n_not_computed": n_nc,
                            "n_cumulative_retained": n_retained,
                            "stage_evaluable_fraction": n_evaluable / n_universe,
                            "stage_pass_fraction": (
                                n_passed / n_evaluable if n_evaluable else np.nan
                            ),
                            "retention_fraction_from_previous": (
                                n_retained / n_previous if n_previous else np.nan
                            ),
                            "retention_fraction_from_universe": (
                                n_retained / n_universe
                            ),
                        }
                    )
                previous = cumulative.copy()
    return pd.DataFrame.from_records(
        records, columns=GATE_ATTRITION_COLUMNS
    ).sort_values(
        [
            "dataset_id",
            "contrast_name",
            "fold_id",
            "stratum_kind",
            "stratum_value",
            "stage_order",
        ],
        kind="stable",
        ignore_index=True,
    )


def _validate_result_tables(
    dataset_id: str,
    *,
    gate_attrition: pd.DataFrame,
    score_geometry: pd.DataFrame,
    candidate_sender_bias: pd.DataFrame,
    resolution_performance: pd.DataFrame,
) -> None:
    required_nonempty = {
        "gate_attrition": gate_attrition,
        "score_geometry": score_geometry,
        "candidate_sender_bias": candidate_sender_bias,
    }
    for name, table in required_nonempty.items():
        if table.empty:
            raise ValueError(f"{name} must not be empty")
    for name, table in (
        *required_nonempty.items(),
        ("resolution_performance", resolution_performance),
    ):
        if not table.empty and set(table["dataset_id"].astype(str)) != {dataset_id}:
            raise ValueError(f"{name} contains a different dataset_id")

    count_columns = (
        "n_universe",
        "n_stage_evaluable",
        "n_stage_passed",
        "n_stage_failed",
        "n_not_estimable",
        "n_not_computed",
        "n_cumulative_retained",
    )
    gate_counts = gate_attrition.loc[:, list(count_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if gate_counts.isna().any(axis=None) or gate_counts.lt(0).any(axis=None):
        raise ValueError("gate attrition counts must be non-negative integers")
    if not np.equal(gate_counts, np.floor(gate_counts)).all(axis=None):
        raise ValueError("gate attrition counts must be integers")
    if not (
        gate_counts["n_stage_evaluable"]
        == gate_counts["n_stage_passed"] + gate_counts["n_stage_failed"]
    ).all():
        raise ValueError("gate evaluable counts do not balance")
    if not (
        gate_counts["n_universe"]
        == gate_counts[
            [
                "n_stage_passed",
                "n_stage_failed",
                "n_not_estimable",
                "n_not_computed",
            ]
        ].sum(axis=1)
    ).all():
        raise ValueError("gate status counts do not cover the universe")
    if gate_counts["n_cumulative_retained"].gt(gate_counts["n_universe"]).any():
        raise ValueError("gate cumulative retention exceeds the universe")
    group_columns = [
        "dataset_id",
        "contrast_name",
        "fold_id",
        "stratum_kind",
        "stratum_value",
    ]
    for _, group in gate_attrition.groupby(group_columns, observed=True, sort=False):
        ordered = group.sort_values("stage_order", kind="stable")
        if tuple(ordered["stage"]) != GATE_STAGES or tuple(
            ordered["stage_order"]
        ) != tuple(range(1, len(GATE_STAGES) + 1)):
            raise ValueError("gate attrition stage axis is incomplete or reordered")
        retained = ordered["n_cumulative_retained"].to_numpy(dtype=int)
        if np.any(np.diff(retained) > 0):
            raise ValueError("gate cumulative retention must be non-increasing")

    geometry_counts = score_geometry.loc[
        :, ["n_rows", "n_finite", "n_missing", "unique_value_count"]
    ].apply(pd.to_numeric, errors="coerce")
    if geometry_counts.isna().any(axis=None) or geometry_counts.lt(0).any(axis=None):
        raise ValueError("score geometry counts must be non-negative")
    if not (
        geometry_counts["n_rows"]
        == geometry_counts["n_finite"] + geometry_counts["n_missing"]
    ).all():
        raise ValueError("score geometry finite/missing counts do not balance")
    if geometry_counts["unique_value_count"].gt(geometry_counts["n_finite"]).any():
        raise ValueError("score geometry unique count exceeds finite rows")

    bias_counts = candidate_sender_bias.loc[
        :,
        [
            "n_parents",
            "n_sender_rows",
            "n_truth_positive_rows",
            "n_truth_negative_rows",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    if bias_counts.isna().any(axis=None) or bias_counts.lt(0).any(axis=None):
        raise ValueError("candidate sender counts must be non-negative")
    if (
        bias_counts["n_truth_positive_rows"] + bias_counts["n_truth_negative_rows"]
        > bias_counts["n_sender_rows"]
    ).any():
        raise ValueError("candidate sender truth counts exceed sender rows")
    if bias_counts["n_parents"].gt(bias_counts["n_sender_rows"]).any():
        raise ValueError("candidate parent count exceeds sender rows")

    if not resolution_performance.empty:
        resolution_counts = resolution_performance.loc[
            :,
            [
                "n_units",
                "n_scored_units",
                "n_truth_positive",
                "n_truth_negative",
            ],
        ].apply(pd.to_numeric, errors="coerce")
        if resolution_counts.isna().any(axis=None) or resolution_counts.lt(0).any(
            axis=None
        ):
            raise ValueError("resolution counts must be non-negative")
        if not (
            resolution_counts["n_units"]
            == resolution_counts["n_truth_positive"]
            + resolution_counts["n_truth_negative"]
        ).all():
            raise ValueError("resolution truth counts do not cover all units")
        if resolution_counts["n_scored_units"].gt(resolution_counts["n_units"]).any():
            raise ValueError("resolution scored count exceeds all units")


@dataclass(frozen=True, slots=True, kw_only=True)
class V7DiagnosticsSpec:
    """Frozen descriptive diagnostic settings."""

    score_heads: tuple[str, ...] = _DEFAULT_SCORE_HEADS
    sender_detection_threshold: float = 0.0
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        heads = tuple(self.score_heads)
        if not heads or len(heads) != len(set(heads)):
            raise ValueError("score_heads must be non-empty and unique")
        unknown = set(heads).difference(_ALLOWED_SCORE_HEADS)
        if unknown:
            raise ValueError(f"unsupported score_heads: {sorted(unknown)}")
        threshold = float(self.sender_detection_threshold)
        if not math.isfinite(threshold):
            raise ValueError("sender_detection_threshold must be finite")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "score_heads", heads)
        object.__setattr__(self, "sender_detection_threshold", threshold)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "v7_diagnostics_spec",
                {
                    "schema_version": self.schema_version,
                    "score_heads": list(heads),
                    "sender_detection_threshold": threshold,
                    "version": V7_DIAGNOSTICS_VERSION,
                },
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class V7DiagnosticsResult:
    """Content-bound gate, geometry, sender-bias, and resolution summaries."""

    dataset_id: str
    crossfit_id: str
    spec: V7DiagnosticsSpec
    gate_attrition: pd.DataFrame
    score_geometry: pd.DataFrame
    candidate_sender_bias: pd.DataFrame
    resolution_performance: pd.DataFrame
    input_digest: str
    output_digest: str = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        dataset = _name(self.dataset_id, field_name="dataset_id")
        crossfit_id = _name(self.crossfit_id, field_name="crossfit_id")
        if not isinstance(self.spec, V7DiagnosticsSpec):
            raise TypeError("spec must be V7DiagnosticsSpec")
        expected = (
            (self.gate_attrition, GATE_ATTRITION_COLUMNS, "gate_attrition"),
            (self.score_geometry, SCORE_GEOMETRY_COLUMNS, "score_geometry"),
            (
                self.candidate_sender_bias,
                CANDIDATE_SENDER_BIAS_COLUMNS,
                "candidate_sender_bias",
            ),
            (
                self.resolution_performance,
                RESOLUTION_PERFORMANCE_COLUMNS,
                "resolution_performance",
            ),
        )
        frozen: dict[str, pd.DataFrame] = {}
        for table, columns, name in expected:
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns:
                raise ValueError(f"{name} columns are invalid")
            frozen[name] = table.copy(deep=True)
        _validate_result_tables(
            dataset,
            gate_attrition=frozen["gate_attrition"],
            score_geometry=frozen["score_geometry"],
            candidate_sender_bias=frozen["candidate_sender_bias"],
            resolution_performance=frozen["resolution_performance"],
        )
        input_digest = _name(self.input_digest, field_name="input_digest")
        if len(input_digest) != 64:
            raise ValueError("input_digest must be a canonical digest")
        output_digest = str(
            canonical_digest(
                {name: _canonical_frame_digest(table) for name, table in frozen.items()}
            )
        )
        result_id = stable_id(
            "v7_diagnostics_result",
            {
                "crossfit_id": crossfit_id,
                "dataset_id": dataset,
                "input_digest": input_digest,
                "output_digest": output_digest,
                "spec_id": self.spec.spec_id,
                "version": V7_DIAGNOSTICS_VERSION,
            },
        )
        object.__setattr__(self, "dataset_id", dataset)
        object.__setattr__(self, "crossfit_id", crossfit_id)
        for name, table in frozen.items():
            object.__setattr__(self, name, table)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "output_digest", output_digest)
        object.__setattr__(self, "result_id", result_id)

    def _require_intact(self) -> None:
        try:
            repeated = V7DiagnosticsResult(
                dataset_id=self.dataset_id,
                crossfit_id=self.crossfit_id,
                spec=self.spec,
                gate_attrition=self.gate_attrition,
                score_geometry=self.score_geometry,
                candidate_sender_bias=self.candidate_sender_bias,
                resolution_performance=self.resolution_performance,
                input_digest=self.input_digest,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("v7 diagnostics result integrity violation") from error
        if repeated.result_id != self.result_id:
            raise ValueError("v7 diagnostics result integrity violation")

    def to_manifest(self) -> dict[str, object]:
        """Return a compact, integrity-checked diagnostic manifest."""

        self._require_intact()
        return {
            "artifact_kind": "v7_diagnostics_result",
            "schema_version": self.spec.schema_version,
            "version": V7_DIAGNOSTICS_VERSION,
            "dataset_id": self.dataset_id,
            "crossfit_id": self.crossfit_id,
            "spec_id": self.spec.spec_id,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "result_id": self.result_id,
            "row_counts": {
                "gate_attrition": len(self.gate_attrition),
                "score_geometry": len(self.score_geometry),
                "candidate_sender_bias": len(self.candidate_sender_bias),
                "resolution_performance": len(self.resolution_performance),
            },
            "formal_inference_allowed": False,
            "unknown_truth_treated_as_negative": False,
            "uncomputed_gate_treated_as_failure": False,
        }


def build_v7_diagnostics(
    crossfit: CrossFitArtifacts,
    *,
    dataset_id: str,
    spec: V7DiagnosticsSpec | None = None,
    truth: pd.DataFrame | None = None,
    cell_counts: pd.DataFrame | None = None,
    gate_annotations: pd.DataFrame | None = None,
    significance: pd.DataFrame | None = None,
    resolution_evaluation: pd.DataFrame | None = None,
) -> V7DiagnosticsResult:
    """Build all preregistered v7 descriptive diagnostics from OOF artifacts."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    dataset = _name(dataset_id, field_name="dataset_id")
    resolved_spec = V7DiagnosticsSpec() if spec is None else spec
    if not isinstance(resolved_spec, V7DiagnosticsSpec):
        raise TypeError("spec must be V7DiagnosticsSpec or None")
    scores, truth_table, count_table = _decorate_scores(
        crossfit, truth=truth, cell_counts=cell_counts
    )
    stage_records = _stage_record_map(
        crossfit,
        gate_annotations=gate_annotations,
        significance=significance,
    )
    stage_statuses = _stage_arrays(scores, stage_records)
    gate_attrition = _gate_attrition(
        scores,
        dataset_id=dataset,
        statuses=stage_statuses,
    )
    geometry = summarize_v7_score_geometry(
        scores,
        dataset_id=dataset,
        score_heads=resolved_spec.score_heads,
    )
    candidate_bias = summarize_v7_candidate_sender_bias(
        scores,
        dataset_id=dataset,
        detection_threshold=resolved_spec.sender_detection_threshold,
    )
    resolution = summarize_v7_resolution_performance(
        resolution_evaluation,
        dataset_id=dataset,
    )
    annotation_digest = _canonical_frame_digest(
        _validate_gate_annotations(
            gate_annotations,
            contrast_names={item.name for item in crossfit.spec.contrasts},
        )
    )
    significance_digest = _canonical_frame_digest(
        _validate_significance(
            significance,
            contrast_names={item.name for item in crossfit.spec.contrasts},
        )
    )
    resolution_digest = _canonical_frame_digest(
        pd.DataFrame() if resolution_evaluation is None else resolution_evaluation
    )
    input_digest = str(
        canonical_digest(
            {
                "cell_counts": _canonical_frame_digest(count_table),
                "crossfit_id": crossfit.crossfit_id,
                "gate_annotations": annotation_digest,
                "resolution_evaluation": resolution_digest,
                "significance": significance_digest,
                "truth": _canonical_frame_digest(truth_table),
            }
        )
    )
    return V7DiagnosticsResult(
        dataset_id=dataset,
        crossfit_id=crossfit.crossfit_id,
        spec=resolved_spec,
        gate_attrition=gate_attrition,
        score_geometry=geometry,
        candidate_sender_bias=candidate_bias,
        resolution_performance=resolution,
        input_digest=input_digest,
    )


__all__ = [
    "CANDIDATE_SENDER_BIAS_COLUMNS",
    "GATE_ATTRITION_COLUMNS",
    "GATE_STAGES",
    "GATE_STAGE_LEDGER_COLUMNS",
    "RESOLUTION_PERFORMANCE_COLUMNS",
    "SCORE_GEOMETRY_COLUMNS",
    "V7_DIAGNOSTICS_VERSION",
    "V7DiagnosticsResult",
    "V7DiagnosticsSpec",
    "build_v7_diagnostics",
    "build_v7_gate_stage_ledger",
    "summarize_v7_resolution_performance",
]
