"""Finalize v7 full-refit distributions without bypassing calibration gates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from crychic.core import canonical_digest, stable_id
from crychic.inference import DifferentialDesignKind

from .v7_full_pipeline_resampling import (
    V7FullPipelineOperation,
    V7FullPipelineResamplingResult,
)

V7_FULL_PIPELINE_INFERENCE_VERSION = "v7_full_refit_inference_finalizer_v2"
V7_FULL_PIPELINE_EFFECT_COLUMNS = (
    "event_id",
    "contrast_name",
    "channel",
    "point_effect",
    "point_standard_error_diagnostic",
    "point_status",
    "point_reason_code",
    "n_bootstrap_total",
    "n_bootstrap_observed",
    "n_permutation_total",
    "n_permutation_observed",
    "n_loso_total",
    "n_loso_observed",
    "diagnostic_bootstrap_standard_error",
    "diagnostic_ci_lower",
    "diagnostic_ci_upper",
    "diagnostic_empirical_p_value",
    "diagnostic_q_value",
    "loso_max_abs_delta",
    "loso_median_abs_delta",
    "loso_sign_agreement",
    "standard_error",
    "ci_lower",
    "ci_upper",
    "p_value",
    "q_value",
    "formal_inference_allowed",
    "formal_inference_status",
    "resampling_result_id",
    "inference_spec_id",
    "calibration_gate_id",
    "source_result_id",
    "result_id",
)
_SCHEMA_VERSION = "1.0.0"
_REQUIRED_CALIBRATION_DESIGNS = tuple(
    sorted(kind.value for kind in DifferentialDesignKind)
)
_CALIBRATION_COLUMNS = (
    "scenario_id",
    "design_kind",
    "n_replicates",
    "empirical_type_i",
    "empirical_fdr",
    "ci_coverage",
)
_GATE_MARKER = "crychic.workflow.v7_full_pipeline_calibration_gate.v1"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _probability(value: object, *, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1]")
    return result


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and math.isnan(float(value))


def _optional(value: object) -> object | None:
    return None if _is_missing_scalar(value) else value


def _canonical_table_digest(table: pd.DataFrame) -> str:
    records: list[dict[str, object]] = []
    for row in table.itertuples(index=False, name=None):
        record: dict[str, object] = {}
        for column, value in zip(table.columns, row, strict=True):
            if _is_missing_scalar(value):
                record[str(column)] = None
            elif isinstance(value, float | np.floating):
                record[str(column)] = {"float_hex": float(value).hex()}
            elif isinstance(value, int | np.integer):
                record[str(column)] = int(value)
            elif isinstance(value, bool | np.bool_):
                record[str(column)] = bool(value)
            else:
                record[str(column)] = str(value)
        records.append(record)
    return str(canonical_digest(records))


@dataclass(frozen=True, slots=True, kw_only=True)
class V7FullPipelineInferenceSpec:
    """Frozen runtime completeness and multiplicity policy."""

    minimum_bootstraps: int = 1_000
    minimum_permutations: int = 1_000
    confidence_level: float = 0.95
    require_complete_loso: bool = True
    fdr_scope: str = "separate_channel_global_event_contrast_bh_v1"
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("minimum_bootstraps", "minimum_permutations"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{field_name} must be an integer >= 2")
        confidence = _probability(
            self.confidence_level,
            field_name="confidence_level",
        )
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")
        if not isinstance(self.require_complete_loso, bool):
            raise TypeError("require_complete_loso must be boolean")
        if self.fdr_scope != "separate_channel_global_event_contrast_bh_v1":
            raise ValueError("unsupported v7 FDR scope")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "v7_full_pipeline_inference_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "fdr_scope": self.fdr_scope,
            "minimum_bootstraps": self.minimum_bootstraps,
            "minimum_permutations": self.minimum_permutations,
            "require_complete_loso": self.require_complete_loso,
            "schema_version": self.schema_version,
            "version": V7_FULL_PIPELINE_INFERENCE_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, init=False)
class V7FullPipelineCalibrationGate:
    """Producer-owned release gate evaluated over all supported design families."""

    metrics: pd.DataFrame
    evidence_digest: str
    required_design_kinds: tuple[str, ...]
    passed: bool
    failure_reasons: tuple[str, ...]
    gate_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "V7FullPipelineCalibrationGate is producer-owned; use "
            "evaluate_v7_full_pipeline_calibration()"
        )

    @classmethod
    def _from_metrics(
        cls,
        metrics: pd.DataFrame,
        *,
        required_design_kinds: tuple[str, ...],
    ) -> V7FullPipelineCalibrationGate:
        table = metrics.copy(deep=True).sort_values(
            ["design_kind", "scenario_id"],
            kind="stable",
            ignore_index=True,
        )
        reasons: list[str] = []
        observed_designs = set(table["design_kind"].astype(str))
        missing_designs = set(required_design_kinds).difference(observed_designs)
        if missing_designs:
            reasons.append("required_design_family_missing")
        if table["n_replicates"].lt(1_000).any():
            reasons.append("null_replicates_below_1000")
        if not table["empirical_type_i"].between(0.035, 0.065).all():
            reasons.append("empirical_type_i_outside_0_035_0_065")
        if table["empirical_fdr"].gt(0.12).any():
            reasons.append("empirical_fdr_exceeds_0_12")
        if not table["ci_coverage"].between(0.92, 0.97).all():
            reasons.append("ci_coverage_outside_0_92_0_97")
        digest = _canonical_table_digest(table)
        self = object.__new__(cls)
        object.__setattr__(self, "metrics", table)
        object.__setattr__(self, "evidence_digest", digest)
        object.__setattr__(self, "required_design_kinds", required_design_kinds)
        object.__setattr__(self, "passed", not reasons)
        object.__setattr__(self, "failure_reasons", tuple(sorted(set(reasons))))
        object.__setattr__(self, "_producer_marker", _GATE_MARKER)
        object.__setattr__(
            self,
            "gate_id",
            stable_id(
                "v7_full_pipeline_calibration_gate",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "evidence_digest": self.evidence_digest,
            "failure_reasons": list(self.failure_reasons),
            "passed": self.passed,
            "required_design_kinds": list(self.required_design_kinds),
            "thresholds": {
                "ci_coverage": [0.92, 0.97],
                "empirical_fdr_maximum": 0.12,
                "empirical_type_i": [0.035, 0.065],
                "minimum_null_replicates": 1_000,
            },
            "version": V7_FULL_PIPELINE_INFERENCE_VERSION,
        }

    def _require_intact(self) -> None:
        if self._producer_marker != _GATE_MARKER:
            raise ValueError("v7 calibration gate producer marker is invalid")
        repeated = evaluate_v7_full_pipeline_calibration(
            self.metrics,
            required_design_kinds=self.required_design_kinds,
        )
        if (
            repeated.evidence_digest != self.evidence_digest
            or repeated.passed != self.passed
            or repeated.failure_reasons != self.failure_reasons
            or repeated.gate_id != self.gate_id
        ):
            raise ValueError("v7 calibration gate integrity validation failed")

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "gate_id": self.gate_id,
            **self._identity_payload(),
            "metrics": self.metrics.to_dict(orient="records"),
        }


def evaluate_v7_full_pipeline_calibration(
    metrics: pd.DataFrame,
    *,
    required_design_kinds: Sequence[str] = _REQUIRED_CALIBRATION_DESIGNS,
) -> V7FullPipelineCalibrationGate:
    """Evaluate the locked type-I/FDR/coverage gate from scenario metrics."""

    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    if tuple(metrics.columns) != _CALIBRATION_COLUMNS:
        raise ValueError("v7 calibration metric columns are invalid")
    if metrics.empty or metrics.isna().any().any():
        raise ValueError("v7 calibration metrics must be non-empty and complete")
    table = metrics.copy(deep=True)
    for column in ("scenario_id", "design_kind"):
        table[column] = table[column].map(
            lambda value, field=column: _name(value, field_name=field)
        )
    if table["scenario_id"].duplicated().any():
        raise ValueError("v7 calibration scenario IDs must be unique")
    table["design_kind"] = table["design_kind"].map(
        lambda value: DifferentialDesignKind(value).value
    )
    replicates = pd.to_numeric(table["n_replicates"], errors="coerce")
    if (
        replicates.isna().any()
        or not replicates.map(lambda value: float(value).is_integer()).all()
        or replicates.lt(1).any()
    ):
        raise ValueError("n_replicates must contain positive integers")
    table["n_replicates"] = replicates.astype(int)
    for column in ("empirical_type_i", "empirical_fdr", "ci_coverage"):
        table[column] = table[column].map(
            lambda value, field=column: _probability(value, field_name=field)
        )
    required = tuple(
        sorted({DifferentialDesignKind(value).value for value in required_design_kinds})
    )
    if not required:
        raise ValueError("required_design_kinds must be non-empty")
    return V7FullPipelineCalibrationGate._from_metrics(
        table,
        required_design_kinds=required,
    )


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.notna()
    effective = numeric.fillna(1.0).to_numpy(dtype=float)
    order = np.argsort(effective, kind="stable")
    adjusted: npt.NDArray[np.float64] = np.empty(len(effective), dtype=float)
    running = 1.0
    for offset in range(len(order) - 1, -1, -1):
        row_index = int(order[offset])
        rank = offset + 1
        running = min(running, effective[row_index] * len(order) / rank, 1.0)
        adjusted[row_index] = running
    result = pd.Series(adjusted, index=values.index, dtype=float)
    result.loc[~observed] = np.nan
    return result


def _point_continuous(
    resampling: V7FullPipelineResamplingResult,
) -> pd.DataFrame:
    table = resampling.point.differential.effects.loc[
        :,
        [
            "event_id",
            "contrast_name",
            "effect",
            "standard_error",
            "status",
            "reason_code",
            "effect_id",
        ],
    ].copy()
    return cast(
        pd.DataFrame,
        table.rename(
            columns={
                "effect": "point_effect",
                "standard_error": "point_standard_error_diagnostic",
                "status": "point_status",
                "reason_code": "point_reason_code",
                "effect_id": "source_result_id",
            }
        ).assign(channel="continuous_raw"),
    )


def _ledger_continuous(
    resampling: V7FullPipelineResamplingResult,
) -> pd.DataFrame:
    ledger = resampling.continuous_effect_ledger()
    if ledger.empty:
        return pd.DataFrame(
            columns=(
                "operation",
                "record_id",
                "event_id",
                "contrast_name",
                "resample_effect",
                "resample_status",
            )
        )
    return ledger.loc[
        :,
        [
            "operation",
            "record_id",
            "event_id",
            "contrast_name",
            "effect",
            "status",
        ],
    ].rename(columns={"effect": "resample_effect", "status": "resample_status"})


def _point_occurrence(
    resampling: V7FullPipelineResamplingResult,
) -> pd.DataFrame:
    occurrence = resampling.point.occurrence
    if occurrence is None:
        return pd.DataFrame()
    continuous = (
        DifferentialDesignKind(resampling.point.spec.design.design_kind)
        is DifferentialDesignKind.CONTINUOUS
    )
    source_column = "log_odds_ratio" if continuous else "prevalence_difference"
    effect_scale = "log_odds" if continuous else "prevalence_difference"
    table = occurrence.occurrence.effects.loc[
        :,
        [
            "event_id",
            "contrast_name",
            source_column,
            "status",
            "reason_code",
            "effect_id",
        ],
    ].copy()
    numeric = pd.to_numeric(table[source_column], errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    table[source_column] = numeric.where(finite, np.nan)
    table["status"] = np.where(finite, "observed", "not_estimable")
    table["reason_code"] = table["reason_code"].where(~finite, None)
    table.loc[
        ~finite & table["reason_code"].isna(),
        "reason_code",
    ] = "occurrence_resampling_effect_not_estimable"
    table["point_standard_error_diagnostic"] = np.nan
    return cast(
        pd.DataFrame,
        table.rename(
            columns={
                source_column: "point_effect",
                "status": "point_status",
                "reason_code": "point_reason_code",
                "effect_id": "source_result_id",
            }
        ).assign(channel=f"occurrence_{effect_scale}"),
    )


def _ledger_occurrence(
    resampling: V7FullPipelineResamplingResult,
) -> pd.DataFrame:
    ledger = resampling.occurrence_effect_ledger()
    if ledger.empty:
        return pd.DataFrame()
    return ledger.loc[
        :,
        [
            "operation",
            "record_id",
            "event_id",
            "contrast_name",
            "occurrence_effect",
            "status",
        ],
    ].rename(
        columns={
            "occurrence_effect": "resample_effect",
            "status": "resample_status",
        }
    )


def _point_hypergraph(
    resampling: V7FullPipelineResamplingResult,
) -> pd.DataFrame:
    hypergraph = resampling.point.hypergraph
    if hypergraph is None:
        return pd.DataFrame()
    table = hypergraph.shrinkage.loc[
        :,
        [
            "edge_id",
            "posterior_effect",
            "posterior_standard_error",
            "status",
            "reason_code",
            "shrinkage_record_id",
        ],
    ].copy()
    table["contrast_name"] = hypergraph.contrast_name
    return cast(
        pd.DataFrame,
        table.rename(
            columns={
                "edge_id": "event_id",
                "posterior_effect": "point_effect",
                "posterior_standard_error": "point_standard_error_diagnostic",
                "status": "point_status",
                "reason_code": "point_reason_code",
                "shrinkage_record_id": "source_result_id",
            }
        ).assign(channel="hypergraph_posterior"),
    )


def _ledger_hypergraph(
    resampling: V7FullPipelineResamplingResult,
) -> pd.DataFrame:
    ledger = resampling.hypergraph_effect_ledger()
    if ledger.empty:
        return pd.DataFrame()
    return ledger.loc[
        :,
        [
            "operation",
            "record_id",
            "edge_id",
            "contrast_name",
            "posterior_effect",
            "status",
        ],
    ].rename(
        columns={
            "edge_id": "event_id",
            "posterior_effect": "resample_effect",
            "status": "resample_status",
        }
    )


def _operation_total(
    resampling: V7FullPipelineResamplingResult,
    operation: V7FullPipelineOperation,
) -> int:
    return sum(record.operation is operation for record in resampling.records)


def _observed_values(
    ledger: pd.DataFrame,
    *,
    event_id: str,
    contrast_name: str,
    operation: V7FullPipelineOperation,
) -> npt.NDArray[np.float64]:
    if ledger.empty:
        return np.asarray([], dtype=np.float64)
    selected = ledger.loc[
        ledger["operation"].eq(operation.value)
        & ledger["event_id"].eq(event_id)
        & ledger["contrast_name"].eq(contrast_name)
    ]
    if selected["record_id"].duplicated().any():
        raise ValueError("v7 resampling ledger has duplicate hypothesis records")
    observed = selected["resample_status"].eq("observed")
    values = pd.to_numeric(
        selected.loc[observed, "resample_effect"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("observed v7 resample effects must be finite")
    return cast(npt.NDArray[np.float64], values)


def _index_observed_values(
    ledger: pd.DataFrame,
) -> dict[tuple[str, str, str], npt.NDArray[np.float64]]:
    if ledger.empty:
        return {}
    key_columns = ["event_id", "contrast_name", "operation"]
    if ledger.duplicated([*key_columns, "record_id"]).any():
        raise ValueError("v7 resampling ledger has duplicate hypothesis records")
    observed = ledger.loc[
        ledger["resample_status"].eq("observed"),
        [*key_columns, "resample_effect"],
    ].copy()
    if observed.empty:
        return {}
    observed["resample_effect"] = pd.to_numeric(
        observed["resample_effect"],
        errors="coerce",
    )
    if (
        observed["resample_effect"].isna().any()
        or np.isinf(observed["resample_effect"]).any()
    ):
        raise ValueError("observed v7 resample effects must be finite")
    return {
        tuple(map(str, key)): group["resample_effect"].to_numpy(dtype=float)
        for key, group in observed.groupby(
            key_columns,
            observed=True,
            sort=False,
        )
    }


def _formal_status(
    *,
    point_observed: bool,
    n_bootstrap_total: int,
    n_bootstrap_observed: int,
    n_permutation_total: int,
    n_permutation_observed: int,
    n_loso_total: int,
    n_loso_observed: int,
    spec: V7FullPipelineInferenceSpec,
    gate: V7FullPipelineCalibrationGate | None,
) -> str:
    if not point_observed:
        return "point_effect_not_estimable"
    if gate is None:
        return "calibration_gate_missing"
    if not gate.passed:
        return "calibration_gate_failed"
    if n_bootstrap_total < spec.minimum_bootstraps:
        return "bootstrap_count_below_minimum"
    if n_bootstrap_observed != n_bootstrap_total:
        return "bootstrap_distribution_incomplete"
    if n_permutation_total < spec.minimum_permutations:
        return "permutation_count_below_minimum"
    if n_permutation_observed != n_permutation_total:
        return "permutation_distribution_incomplete"
    if spec.require_complete_loso and (
        n_loso_total < 1 or n_loso_observed != n_loso_total
    ):
        return "loso_distribution_incomplete"
    return "released"


def _finalize_channel(
    point: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    resampling: V7FullPipelineResamplingResult,
    spec: V7FullPipelineInferenceSpec,
    gate: V7FullPipelineCalibrationGate | None,
) -> pd.DataFrame:
    if point.empty:
        return pd.DataFrame(columns=V7_FULL_PIPELINE_EFFECT_COLUMNS)
    point = point.sort_values(
        ["event_id", "contrast_name"],
        kind="stable",
        ignore_index=True,
    )
    if point.duplicated(["event_id", "contrast_name"]).any():
        raise ValueError("v7 point hypothesis keys must be unique within channel")
    point_keys = set(
        point.loc[:, ["event_id", "contrast_name"]].itertuples(
            index=False,
            name=None,
        )
    )
    if not ledger.empty:
        ledger_keys = set(
            ledger.loc[:, ["event_id", "contrast_name"]].itertuples(
                index=False,
                name=None,
            )
        )
        extras = ledger_keys.difference(point_keys)
        if extras:
            raise ValueError(
                "v7 resample ledger contains hypotheses outside point axis"
            )
    totals = {
        operation: _operation_total(resampling, operation)
        for operation in V7FullPipelineOperation
    }
    observed_values = _index_observed_values(ledger)
    empty_values = np.asarray([], dtype=np.float64)
    alpha = 0.5 * (1.0 - spec.confidence_level)
    rows: list[dict[str, object]] = []
    point_records = cast(list[dict[str, object]], point.to_dict(orient="records"))
    for point_row in point_records:
        event_id = str(point_row["event_id"])
        contrast_name = str(point_row["contrast_name"])
        point_value_raw = _optional(point_row["point_effect"])
        point_observed = (
            point_row["point_status"] == "observed"
            and point_value_raw is not None
            and math.isfinite(_finite(point_value_raw, field_name="point_effect"))
        )
        point_value = (
            _finite(point_value_raw, field_name="point_effect")
            if point_observed
            else None
        )
        bootstrap = observed_values.get(
            (
                event_id,
                contrast_name,
                V7FullPipelineOperation.SUBJECT_BOOTSTRAP.value,
            ),
            empty_values,
        )
        permutation = observed_values.get(
            (
                event_id,
                contrast_name,
                V7FullPipelineOperation.CONDITION_PERMUTATION.value,
            ),
            empty_values,
        )
        loso = observed_values.get(
            (
                event_id,
                contrast_name,
                V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT.value,
            ),
            empty_values,
        )
        diagnostic_se = (
            float(np.std(bootstrap, ddof=1)) if len(bootstrap) >= 2 else None
        )
        diagnostic_lower: float | None = None
        diagnostic_upper: float | None = None
        if len(bootstrap) >= 2:
            quantiles = np.asarray(
                np.quantile(
                    bootstrap,
                    [alpha, 1.0 - alpha],
                    method="linear",
                ),
                dtype=float,
            )
            diagnostic_lower = float(quantiles[0])
            diagnostic_upper = float(quantiles[1])
        diagnostic_p = None
        if point_value is not None and len(permutation):
            diagnostic_p = float(
                (1 + np.count_nonzero(np.abs(permutation) >= abs(point_value)))
                / (len(permutation) + 1)
            )
        loso_delta = (
            np.abs(loso - point_value)
            if point_value is not None and len(loso)
            else np.asarray([], dtype=float)
        )
        formal_status = _formal_status(
            point_observed=point_observed,
            n_bootstrap_total=totals[V7FullPipelineOperation.SUBJECT_BOOTSTRAP],
            n_bootstrap_observed=len(bootstrap),
            n_permutation_total=totals[V7FullPipelineOperation.CONDITION_PERMUTATION],
            n_permutation_observed=len(permutation),
            n_loso_total=totals[V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT],
            n_loso_observed=len(loso),
            spec=spec,
            gate=gate,
        )
        released = formal_status == "released"
        rows.append(
            {
                **point_row,
                "n_bootstrap_total": totals[V7FullPipelineOperation.SUBJECT_BOOTSTRAP],
                "n_bootstrap_observed": len(bootstrap),
                "n_permutation_total": totals[
                    V7FullPipelineOperation.CONDITION_PERMUTATION
                ],
                "n_permutation_observed": len(permutation),
                "n_loso_total": totals[V7FullPipelineOperation.LEAVE_ONE_SUBJECT_OUT],
                "n_loso_observed": len(loso),
                "diagnostic_bootstrap_standard_error": diagnostic_se,
                "diagnostic_ci_lower": diagnostic_lower,
                "diagnostic_ci_upper": diagnostic_upper,
                "diagnostic_empirical_p_value": diagnostic_p,
                "diagnostic_q_value": None,
                "loso_max_abs_delta": (
                    None if not len(loso_delta) else float(loso_delta.max())
                ),
                "loso_median_abs_delta": (
                    None if not len(loso_delta) else float(np.median(loso_delta))
                ),
                "loso_sign_agreement": (
                    None
                    if point_value is None or not len(loso)
                    else float(np.mean(np.sign(loso) == np.sign(point_value)))
                ),
                "standard_error": diagnostic_se if released else None,
                "ci_lower": diagnostic_lower if released else None,
                "ci_upper": diagnostic_upper if released else None,
                "p_value": diagnostic_p if released else None,
                "q_value": None,
                "formal_inference_allowed": released,
                "formal_inference_status": formal_status,
                "resampling_result_id": resampling.result_id,
                "inference_spec_id": spec.spec_id,
                "calibration_gate_id": None if gate is None else gate.gate_id,
                "result_id": None,
            }
        )
    result = pd.DataFrame(rows)
    result["diagnostic_q_value"] = _bh(result["diagnostic_empirical_p_value"])
    formal_input = result["p_value"].where(result["formal_inference_allowed"])
    result["q_value"] = _bh(formal_input)
    result.loc[~result["formal_inference_allowed"], "q_value"] = np.nan
    result["result_id"] = [
        stable_id(
            "v7_full_pipeline_effect_result",
            {
                "calibration_gate_id": row.calibration_gate_id,
                "channel": row.channel,
                "contrast_name": row.contrast_name,
                "diagnostic_ci_lower": _optional(row.diagnostic_ci_lower),
                "diagnostic_ci_upper": _optional(row.diagnostic_ci_upper),
                "diagnostic_empirical_p_value": _optional(
                    row.diagnostic_empirical_p_value
                ),
                "diagnostic_q_value": _optional(row.diagnostic_q_value),
                "event_id": row.event_id,
                "formal_inference_status": row.formal_inference_status,
                "inference_spec_id": spec.spec_id,
                "point_effect": _optional(row.point_effect),
                "q_value": _optional(row.q_value),
                "resampling_result_id": resampling.result_id,
                "source_result_id": row.source_result_id,
            },
            schema_version=_SCHEMA_VERSION,
        )
        for row in result.itertuples(index=False)
    ]
    return result.loc[
        :,
        list(V7_FULL_PIPELINE_EFFECT_COLUMNS),
    ].copy()


@dataclass(frozen=True, slots=True, kw_only=True)
class V7FullPipelineInferenceResult:
    """Separate continuous, occurrence, and hypergraph full-refit channels."""

    continuous_effects: pd.DataFrame
    occurrence_effects: pd.DataFrame
    hypergraph_effects: pd.DataFrame
    spec: V7FullPipelineInferenceSpec
    resampling_result_id: str
    calibration_gate: V7FullPipelineCalibrationGate | None
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, V7FullPipelineInferenceSpec):
            raise TypeError("spec must be V7FullPipelineInferenceSpec")
        resampling_id = _name(
            self.resampling_result_id,
            field_name="resampling_result_id",
        )
        gate = self.calibration_gate
        if gate is not None and not isinstance(gate, V7FullPipelineCalibrationGate):
            raise TypeError("calibration_gate must be typed or None")
        if gate is not None:
            gate._require_intact()
        tables: dict[str, pd.DataFrame] = {}
        for field_name in (
            "continuous_effects",
            "occurrence_effects",
            "hypergraph_effects",
        ):
            table = getattr(self, field_name)
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != (
                V7_FULL_PIPELINE_EFFECT_COLUMNS
            ):
                raise ValueError(f"{field_name} has invalid columns")
            if table["result_id"].duplicated().any():
                raise ValueError(f"{field_name} result IDs must be unique")
            unreleased = ~table["formal_inference_allowed"]
            if (
                table.loc[
                    unreleased,
                    ["standard_error", "ci_lower", "ci_upper", "p_value", "q_value"],
                ]
                .notna()
                .any(axis=None)
            ):
                raise ValueError("unreleased v7 effects cannot expose formal fields")
            tables[field_name] = table.copy(deep=True)
        for field_name, table in tables.items():
            object.__setattr__(self, field_name, table)
        object.__setattr__(self, "resampling_result_id", resampling_id)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "v7_full_pipeline_inference_result",
                {
                    "calibration_gate_id": None if gate is None else gate.gate_id,
                    "continuous_result_ids": list(
                        tables["continuous_effects"]["result_id"]
                    ),
                    "hypergraph_result_ids": list(
                        tables["hypergraph_effects"]["result_id"]
                    ),
                    "inference_spec_id": self.spec.spec_id,
                    "occurrence_result_ids": list(
                        tables["occurrence_effects"]["result_id"]
                    ),
                    "resampling_result_id": resampling_id,
                    "version": V7_FULL_PIPELINE_INFERENCE_VERSION,
                },
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "version": V7_FULL_PIPELINE_INFERENCE_VERSION,
            "resampling_result_id": self.resampling_result_id,
            "spec": self.spec.to_dict(),
            "calibration_gate": (
                None
                if self.calibration_gate is None
                else self.calibration_gate.to_dict()
            ),
            "channel_rows": {
                "continuous_raw": len(self.continuous_effects),
                "occurrence_effect": len(self.occurrence_effects),
                "hypergraph_posterior": len(self.hypergraph_effects),
            },
            "formal_rows": sum(
                int(table["formal_inference_allowed"].sum())
                for table in (
                    self.continuous_effects,
                    self.occurrence_effects,
                    self.hypergraph_effects,
                )
            ),
        }


def finalize_v7_full_pipeline_inference(
    resampling: V7FullPipelineResamplingResult,
    *,
    spec: V7FullPipelineInferenceSpec | None = None,
    calibration_gate: V7FullPipelineCalibrationGate | None = None,
) -> V7FullPipelineInferenceResult:
    """Finalize candidate distributions and release fields only after all gates."""

    if not isinstance(resampling, V7FullPipelineResamplingResult):
        raise TypeError("resampling must be V7FullPipelineResamplingResult")
    resampling._require_intact()
    resolved = spec or V7FullPipelineInferenceSpec()
    if not isinstance(resolved, V7FullPipelineInferenceSpec):
        raise TypeError("spec must be V7FullPipelineInferenceSpec or None")
    if calibration_gate is not None and not isinstance(
        calibration_gate,
        V7FullPipelineCalibrationGate,
    ):
        raise TypeError("calibration_gate must be typed or None")
    if calibration_gate is not None:
        calibration_gate._require_intact()
    continuous = _finalize_channel(
        _point_continuous(resampling),
        _ledger_continuous(resampling),
        resampling=resampling,
        spec=resolved,
        gate=calibration_gate,
    )
    occurrence = _finalize_channel(
        _point_occurrence(resampling),
        _ledger_occurrence(resampling),
        resampling=resampling,
        spec=resolved,
        gate=calibration_gate,
    )
    hypergraph = _finalize_channel(
        _point_hypergraph(resampling),
        _ledger_hypergraph(resampling),
        resampling=resampling,
        spec=resolved,
        gate=calibration_gate,
    )
    return V7FullPipelineInferenceResult(
        continuous_effects=continuous,
        occurrence_effects=occurrence,
        hypergraph_effects=hypergraph,
        spec=resolved,
        resampling_result_id=resampling.result_id,
        calibration_gate=calibration_gate,
    )


__all__ = [
    "V7_FULL_PIPELINE_EFFECT_COLUMNS",
    "V7_FULL_PIPELINE_INFERENCE_VERSION",
    "V7FullPipelineCalibrationGate",
    "V7FullPipelineInferenceResult",
    "V7FullPipelineInferenceSpec",
    "evaluate_v7_full_pipeline_calibration",
    "finalize_v7_full_pipeline_inference",
]
