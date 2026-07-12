"""Hierarchical comparison coverage and pre-selection risk curves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from crychic.core import stable_id

COVERAGE_STAGES = (
    "resource_mapped",
    "sample_supported",
    "state_eligible",
    "receptor_eligible",
    "target_basis_eligible",
    "family_estimable",
    "lr_identifiable",
    "sender_estimable",
    "inferential_eligible",
)
COVERAGE_STATUSES = frozenset(
    {"eligible", "ineligible", "not_estimable", "missing", "failed"}
)
RISK_METRICS = (
    "synthetic_auprc",
    "ligand_only_false_activation",
    "loso_stability",
    "supportive_biology_direction",
)

LEDGER_COLUMNS = (
    "dataset",
    "method",
    "gate_threshold",
    "comparison_id",
    "stage",
    "stage_index",
    "stage_status",
    "stage_reason_code",
    "cumulative_eligible",
    "first_failure_stage",
    "first_failure_reason_code",
)


def _required_input_columns() -> set[str]:
    return {
        "dataset",
        "method",
        "gate_threshold",
        "comparison_id",
        *COVERAGE_STAGES,
        *(f"{stage}_reason_code" for stage in COVERAGE_STAGES),
    }


def _validate_stage_rows(table: pd.DataFrame) -> None:
    for stage in COVERAGE_STAGES:
        status = table[stage].astype("string")
        invalid = set(status.dropna().astype(str)).difference(COVERAGE_STATUSES)
        if invalid or status.isna().any():
            raise ValueError(
                f"coverage stage {stage!r} has invalid statuses: {sorted(invalid)}"
            )
        reason = table[f"{stage}_reason_code"]
        eligible = status.eq("eligible")
        if (eligible & reason.notna()).any():
            raise ValueError(f"eligible stage {stage!r} must not carry a reason")
        if ((~eligible) & reason.isna()).any():
            raise ValueError(f"non-eligible stage {stage!r} requires a reason")


def _validate_fixed_universe(table: pd.DataFrame) -> None:
    for (dataset, method), group in table.groupby(
        ["dataset", "method"], sort=False, observed=True
    ):
        universes = [
            frozenset(threshold_rows["comparison_id"].astype(str))
            for _, threshold_rows in group.groupby(
                "gate_threshold", sort=True, observed=True
            )
        ]
        if not universes or any(universe != universes[0] for universe in universes):
            raise ValueError(
                "coverage thresholds must use one fixed comparison universe for "
                f"dataset={dataset!r}, method={method!r}"
            )


def _validate_hierarchy(table: pd.DataFrame) -> None:
    for row in table.itertuples(index=False):
        failed = False
        for stage in COVERAGE_STAGES:
            status = str(getattr(row, stage))
            if failed and status == "eligible":
                raise ValueError(
                    f"comparison {row.comparison_id!r} becomes eligible again at "
                    f"stage {stage!r}"
                )
            failed = failed or status != "eligible"


def _validate_threshold_monotonicity(ledger: pd.DataFrame) -> None:
    grouped = ledger.groupby(
        ["dataset", "method", "comparison_id", "stage"],
        sort=False,
        observed=True,
    )
    for key, group in grouped:
        ordered = group.sort_values("gate_threshold", kind="stable")
        values = ordered["cumulative_eligible"].astype(int).to_numpy()
        if len(values) > 1 and (values[1:] > values[:-1]).any():
            raise ValueError(
                f"higher gate threshold reactivates an ineligible comparison: {key!r}"
            )


def build_coverage_ledger(records: pd.DataFrame) -> pd.DataFrame:
    """Expand wide stage statuses into an auditable cumulative coverage ledger."""

    missing = _required_input_columns().difference(records.columns)
    if missing:
        raise ValueError(f"coverage records are missing columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("coverage records must not be empty")
    table = records.loc[:, sorted(_required_input_columns())].copy(deep=True)
    identifier_columns = ("dataset", "method", "comparison_id")
    if table.loc[:, list(identifier_columns)].isna().any().any():
        raise ValueError("coverage identifiers must not contain missing values")
    for column in identifier_columns:
        if any(not str(value) for value in table[column]):
            raise ValueError(f"coverage identifier {column!r} must not be empty")
        table[column] = table[column].astype(str)
    threshold = pd.to_numeric(table["gate_threshold"], errors="coerce")
    if threshold.isna().any() or any(not math.isfinite(value) for value in threshold):
        raise ValueError("gate_threshold must contain finite numeric values")
    table["gate_threshold"] = threshold.astype(float)
    key = ["dataset", "method", "gate_threshold", "comparison_id"]
    if table.duplicated(key).any():
        raise ValueError("coverage records must have unique threshold comparisons")
    _validate_stage_rows(table)
    _validate_fixed_universe(table)
    _validate_hierarchy(table)

    rows: list[dict[str, object]] = []
    for record in table.to_dict(orient="records"):
        cumulative = True
        first_failure_stage: str | None = None
        first_failure_reason: str | None = None
        for stage_index, stage in enumerate(COVERAGE_STAGES):
            status = str(record[stage])
            reason_value = record[f"{stage}_reason_code"]
            reason = None if pd.isna(reason_value) else str(reason_value)
            cumulative = cumulative and status == "eligible"
            if not cumulative and first_failure_stage is None:
                first_failure_stage = stage
                first_failure_reason = reason
            rows.append(
                {
                    "dataset": record["dataset"],
                    "method": record["method"],
                    "gate_threshold": float(record["gate_threshold"]),
                    "comparison_id": record["comparison_id"],
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_status": status,
                    "stage_reason_code": reason,
                    "cumulative_eligible": cumulative,
                    "first_failure_stage": first_failure_stage,
                    "first_failure_reason_code": first_failure_reason,
                }
            )
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    _validate_threshold_monotonicity(ledger)
    return ledger.sort_values(
        ["dataset", "method", "gate_threshold", "comparison_id", "stage_index"],
        kind="stable",
        ignore_index=True,
    )


def summarize_coverage_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize cumulative comparison coverage with fixed denominators."""

    missing = set(LEDGER_COLUMNS).difference(ledger.columns)
    if missing:
        raise ValueError(f"coverage ledger is missing columns: {sorted(missing)}")
    if ledger.empty:
        raise ValueError("coverage ledger must not be empty")
    group_columns = ["dataset", "method", "gate_threshold", "stage", "stage_index"]
    rows: list[dict[str, object]] = []
    for key, group in ledger.groupby(group_columns, sort=False, observed=True):
        dataset, method, threshold_raw, stage, stage_index_raw = cast(
            tuple[object, object, object, object, object], key
        )
        threshold = float(cast(float, threshold_raw))
        stage_index = int(cast(int, stage_index_raw))
        total = int(group["comparison_id"].nunique())
        eligible = int(group["cumulative_eligible"].sum())
        status_counts = group["stage_status"].value_counts().to_dict()
        rows.append(
            {
                "dataset": str(dataset),
                "method": str(method),
                "gate_threshold": float(threshold),
                "stage": str(stage),
                "stage_index": int(stage_index),
                "n_comparisons": total,
                "n_cumulative_eligible": eligible,
                "comparison_coverage": eligible / total,
                **{
                    f"n_{status}": int(status_counts.get(status, 0))
                    for status in sorted(COVERAGE_STATUSES)
                },
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "method", "gate_threshold", "stage_index"],
        kind="stable",
        ignore_index=True,
    )


def _validate_risk_metrics(table: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "method", "gate_threshold"}
    for metric in RISK_METRICS:
        required.update({metric, f"{metric}_status", f"{metric}_reason_code"})
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"risk metrics are missing columns: {sorted(missing)}")
    result = table.loc[:, sorted(required)].copy(deep=True)
    key = ["dataset", "method", "gate_threshold"]
    if result.duplicated(key).any():
        raise ValueError("risk metrics must have one row per method threshold")
    for metric in RISK_METRICS:
        status = result[f"{metric}_status"].astype(str)
        invalid = set(status).difference({"observed", "not_estimable"})
        if invalid:
            raise ValueError(f"risk metric {metric!r} has invalid statuses")
        numeric = pd.to_numeric(result[metric], errors="coerce")
        observed = status.eq("observed")
        reason = result[f"{metric}_reason_code"]
        if (observed & (numeric.isna() | reason.notna())).any():
            raise ValueError(
                f"observed risk metric {metric!r} requires value and no reason"
            )
        if ((~observed) & (numeric.notna() | reason.isna())).any():
            raise ValueError(
                f"not-estimable risk metric {metric!r} requires NA and reason"
            )
        result[metric] = numeric
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageRiskCurve:
    """Coverage-risk points only; threshold selection is deliberately separate."""

    table: pd.DataFrame
    coverage_stage: str
    selection_allowed: bool = False
    curve_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.coverage_stage not in COVERAGE_STAGES:
            raise ValueError("coverage_stage must name a registered coverage layer")
        if self.selection_allowed:
            raise ValueError("coverage-risk construction must not select a threshold")
        table = self.table.copy(deep=True)
        if table.empty:
            raise ValueError("coverage-risk curve must not be empty")
        payload = {
            "coverage_stage": self.coverage_stage,
            "records_json": table.sort_values(
                ["dataset", "method", "gate_threshold"], kind="stable"
            ).to_json(orient="records", double_precision=15),
            "selection_allowed": False,
        }
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "curve_id", stable_id("coverage_risk_curve", payload))


def build_coverage_risk_curve(
    ledger: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    *,
    coverage_stage: str = "lr_identifiable",
) -> CoverageRiskCurve:
    """Join one cumulative coverage layer to preregistered reliability metrics."""

    if coverage_stage not in COVERAGE_STAGES:
        raise ValueError("coverage_stage must name a registered coverage layer")
    coverage = summarize_coverage_ledger(ledger)
    coverage = coverage.loc[coverage["stage"].eq(coverage_stage)].copy()
    risks = _validate_risk_metrics(risk_metrics)
    key = ["dataset", "method", "gate_threshold"]
    merged = coverage.merge(
        risks,
        how="outer",
        on=key,
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("coverage and risk thresholds must match exactly")
    merged = merged.drop(columns="_merge")
    unavailable: list[str] = []
    for _, row in merged.iterrows():
        missing_metrics = [
            metric for metric in RISK_METRICS if row[f"{metric}_status"] != "observed"
        ]
        unavailable.append(
            ""
            if not missing_metrics
            else "risk_metric_not_estimable:" + ",".join(missing_metrics)
        )
    merged["curve_status"] = [
        "observed" if not reason else "not_estimable" for reason in unavailable
    ]
    merged["curve_reason_code"] = [reason or None for reason in unavailable]
    merged["threshold_selected"] = False
    merged["selection_policy"] = "curve_only_pre_holdout_selection_required"
    return CoverageRiskCurve(table=merged, coverage_stage=coverage_stage)


__all__ = [
    "COVERAGE_STAGES",
    "COVERAGE_STATUSES",
    "RISK_METRICS",
    "CoverageRiskCurve",
    "build_coverage_ledger",
    "build_coverage_risk_curve",
    "summarize_coverage_ledger",
]
