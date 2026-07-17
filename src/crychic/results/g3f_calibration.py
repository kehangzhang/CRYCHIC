"""Independent, replay-verifiable persistence for ADR-012 G3-F campaigns.

The replicate registry and per-hypothesis ledger are authoritative.  Loading
an artifact reconstructs the frozen universe, hierarchical procedure,
protocol, and every replicate before recomputing scenario metrics,
permutation diagnostics, evidence, and the release gate.  Persisted summaries
therefore cannot authorize formal inference on their own.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, SeedLineage, canonical_json, stable_id
from crychic.inference.calibration_attestation import (
    CalibrationGeneratorManifest,
    CalibrationReplayRegistry,
)
from crychic.inference.g3f_calibration import (
    G3FrequencyCalibrationCampaign,
    G3FrequencyCalibrationGate,
    G3FrequencyCalibrationProtocol,
    G3FrequencyCalibrationReplicate,
    G3FrequencyDependenceStructure,
    G3FrequencyReplicateStatus,
    build_g3_frequency_calibration_gate,
    summarize_attested_g3_frequency_calibration_campaign,
    summarize_g3_frequency_calibration_campaign,
)
from crychic.inference.hierarchical import (
    freeze_hierarchical_fdr_spec,
)
from crychic.inference.hypotheses import (
    HypothesisDeclaration,
    HypothesisRole,
    freeze_hypothesis_universe,
)

from ._schema import load_schema_document
from .errors import IncompleteResultError, ResultValidationError, ResultWriteError

G3F_CALIBRATION_RESULT_SCHEMA_VERSION = "1.0.0"
G3F_CALIBRATION_ARTIFACT_KIND = "adr012_g3f_calibration_result"

G3F_CALIBRATION_REPLICATE_TABLE = "g3f_calibration_replicates"
G3F_CALIBRATION_HYPOTHESIS_TABLE = "g3f_calibration_hypotheses"
G3F_CALIBRATION_SCENARIO_TABLE = "g3f_calibration_scenarios"
G3F_CALIBRATION_PERMUTATION_TABLE = "g3f_calibration_permutation_diagnostics"

_CAMPAIGN_SCHEMA_VERSION = "2.0.0"
_TABLE_SCHEMA_VERSION = "1.0.0"
_STATUS_FILENAME = "_status.json"
_MANIFEST_FILENAME = "g3f_calibration_manifest.json"
_COMPLETE = "complete"
_INCOMPLETE = "incomplete"
_METRIC_WEIGHTING = (
    "hypothesis_equal_within_replicate_then_informative_replicate_equal_v2"
)

_SCHEMA_FILES = {
    G3F_CALIBRATION_REPLICATE_TABLE: "g3f_calibration_replicates.schema.json",
    G3F_CALIBRATION_HYPOTHESIS_TABLE: "g3f_calibration_hypotheses.schema.json",
    G3F_CALIBRATION_SCENARIO_TABLE: "g3f_calibration_scenarios.schema.json",
    G3F_CALIBRATION_PERMUTATION_TABLE: (
        "g3f_calibration_permutation_diagnostics.schema.json"
    ),
}


def _read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
    reader = cast(Callable[..., pd.DataFrame], pd.read_parquet)
    return reader(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class _ColumnContract:
    dtype: str
    nullable: bool
    enum: tuple[object, ...]
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class _TableContract:
    name: str
    filename: str
    schema_filename: str
    primary_key: tuple[str, ...]
    logical_key: tuple[str, ...]
    columns: Mapping[str, _ColumnContract]


def _fail(
    message: str,
    *,
    code: str,
    field_name: str,
    remediation: str = "Reject the artifact and regenerate it from intact ledgers",
) -> NoReturn:
    raise ResultValidationError(
        message,
        code=code,
        field=field_name,
        remediation=remediation,
    )


def _const(properties: Mapping[str, Any], name: str, *, schema: str) -> Any:
    value = properties.get(name)
    if not isinstance(value, Mapping) or "const" not in value:
        _fail(
            f"G3-F schema {schema!r} lacks {name!r}",
            code="invalid_g3f_calibration_schema",
            field_name=name,
            remediation="Restore the released G3-F calibration schemas",
        )
    return value["const"]


@cache
def _table_contract(name: str) -> _TableContract:
    if name not in _SCHEMA_FILES:
        raise KeyError(name)
    schema_filename = _SCHEMA_FILES[name]
    document = load_schema_document(schema_filename)
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        _fail(
            f"G3-F schema {schema_filename!r} has no properties",
            code="invalid_g3f_calibration_schema",
            field_name="properties",
        )
    version = _const(properties, "artifact_schema_version", schema=schema_filename)
    table_name = _const(properties, "table", schema=schema_filename)
    filename = _const(properties, "filename", schema=schema_filename)
    primary_key = _const(properties, "primary_key", schema=schema_filename)
    logical_key = _const(properties, "logical_key", schema=schema_filename)
    _const(properties, "foreign_keys", schema=schema_filename)
    columns_block = properties.get("columns")
    raw_columns = (
        columns_block.get("properties")
        if isinstance(columns_block, Mapping)
        else None
    )
    if (
        version != _TABLE_SCHEMA_VERSION
        or table_name != name
        or not isinstance(filename, str)
        or not isinstance(primary_key, list)
        or not all(isinstance(item, str) for item in primary_key)
        or not isinstance(logical_key, list)
        or not all(isinstance(item, str) for item in logical_key)
        or not isinstance(raw_columns, Mapping)
    ):
        _fail(
            f"G3-F schema {schema_filename!r} is incompatible",
            code="invalid_g3f_calibration_schema",
            field_name="artifact_schema_version",
        )
    columns: dict[str, _ColumnContract] = {}
    for column_name, raw in raw_columns.items():
        if not isinstance(column_name, str) or not isinstance(raw, Mapping):
            _fail(
                "G3-F table has an invalid column contract",
                code="invalid_g3f_calibration_schema",
                field_name="columns",
            )
        dtype = raw.get("dtype")
        nullable = raw.get("nullable")
        enum = raw.get("enum", [])
        minimum = raw.get("minimum")
        maximum = raw.get("maximum")
        if (
            dtype not in {"string", "float", "integer", "boolean"}
            or not isinstance(nullable, bool)
            or not isinstance(enum, list)
            or (
                minimum is not None
                and (isinstance(minimum, bool) or not isinstance(minimum, numbers.Real))
            )
            or (
                maximum is not None
                and (isinstance(maximum, bool) or not isinstance(maximum, numbers.Real))
            )
        ):
            _fail(
                f"G3-F column {column_name!r} has an invalid contract",
                code="invalid_g3f_calibration_schema",
                field_name=column_name,
            )
        columns[column_name] = _ColumnContract(
            dtype=cast(str, dtype),
            nullable=nullable,
            enum=tuple(enum),
            minimum=None if minimum is None else float(minimum),
            maximum=None if maximum is None else float(maximum),
        )
    keys = tuple(cast(list[str], primary_key)) + tuple(cast(list[str], logical_key))
    if not set(keys).issubset(columns):
        _fail(
            "G3-F table key is absent from its columns",
            code="invalid_g3f_calibration_schema",
            field_name="primary_key,logical_key",
        )
    return _TableContract(
        name=name,
        filename=filename,
        schema_filename=schema_filename,
        primary_key=tuple(cast(list[str], primary_key)),
        logical_key=tuple(cast(list[str], logical_key)),
        columns=columns,
    )


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(cast(Any, value)))
    except (TypeError, ValueError):
        return False


def _validate_scalar(
    value: object,
    contract: _ColumnContract,
    *,
    table_name: str,
    column_name: str,
) -> None:
    if _is_missing(value):
        if not contract.nullable:
            _fail(
                f"{table_name}.{column_name} cannot be null",
                code="invalid_g3f_calibration_null",
                field_name=column_name,
            )
        return
    if contract.dtype == "string":
        valid = isinstance(value, str) and bool(value) and value == value.strip()
    elif contract.dtype == "boolean":
        valid = isinstance(value, (bool, np.bool_))
    elif contract.dtype == "integer":
        valid = isinstance(value, numbers.Integral) and not isinstance(
            value, (bool, np.bool_)
        )
    else:
        valid = isinstance(value, numbers.Real) and not isinstance(
            value, (bool, np.bool_)
        )
        if valid:
            valid = math.isfinite(float(cast(numbers.Real, value)))
    if not valid:
        _fail(
            f"{table_name}.{column_name} violates dtype {contract.dtype}",
            code="invalid_g3f_calibration_dtype",
            field_name=column_name,
        )
    if contract.enum and value not in contract.enum:
        _fail(
            f"{table_name}.{column_name} is outside its enum",
            code="invalid_g3f_calibration_enum",
            field_name=column_name,
        )
    if contract.dtype in {"integer", "float"}:
        numeric = float(cast(numbers.Real, value))
        if contract.minimum is not None and numeric < contract.minimum:
            _fail(
                f"{table_name}.{column_name} is below its minimum",
                code="invalid_g3f_calibration_range",
                field_name=column_name,
            )
        if contract.maximum is not None and numeric > contract.maximum:
            _fail(
                f"{table_name}.{column_name} exceeds its maximum",
                code="invalid_g3f_calibration_range",
                field_name=column_name,
            )


def _validate_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    contract = _table_contract(name)
    if tuple(frame.columns) != tuple(contract.columns):
        _fail(
            f"G3-F table {name!r} columns do not match its schema",
            code="invalid_g3f_calibration_columns",
            field_name=name,
        )
    if frame.duplicated(list(contract.primary_key)).any():
        _fail(
            f"G3-F table {name!r} has duplicate primary keys",
            code="duplicate_g3f_calibration_primary_key",
            field_name=",".join(contract.primary_key),
        )
    if frame.duplicated(list(contract.logical_key)).any():
        _fail(
            f"G3-F table {name!r} has duplicate logical keys",
            code="duplicate_g3f_calibration_logical_key",
            field_name=",".join(contract.logical_key),
        )
    for column_name, column_contract in contract.columns.items():
        for value in frame[column_name].array:
            _validate_scalar(
                value,
                column_contract,
                table_name=name,
                column_name=column_name,
            )
    return frame


def _hypothesis_row_id(
    *,
    replicate_id: str,
    protocol_id: str,
    scenario_id: str,
    hypothesis_index: int,
    hypothesis_id: str,
    hypothesis_role: str,
    parent_hypothesis_id: str | None,
    truth_non_null: bool,
    candidate_rejected_at_alpha: bool,
    baseline_rejected_at_alpha: bool,
    ci_contains_truth: bool,
    candidate_ci_interval_width: float | None,
    baseline_ci_interval_width: float | None,
) -> str:
    return stable_id(
        "g3f_calibration_hypothesis_row",
        {
            "replicate_id": replicate_id,
            "protocol_id": protocol_id,
            "scenario_id": scenario_id,
            "hypothesis_index": hypothesis_index,
            "hypothesis_id": hypothesis_id,
            "hypothesis_role": hypothesis_role,
            "parent_hypothesis_id": parent_hypothesis_id,
            "truth_non_null": truth_non_null,
            "candidate_rejected_at_alpha": candidate_rejected_at_alpha,
            "baseline_rejected_at_alpha": baseline_rejected_at_alpha,
            "ci_contains_truth": ci_contains_truth,
            "candidate_ci_interval_width": candidate_ci_interval_width,
            "baseline_ci_interval_width": baseline_ci_interval_width,
        },
        schema_version="1",
    )


def _empty_or_frame(name: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    contract = _table_contract(name)
    frame = pd.DataFrame(rows, columns=tuple(contract.columns))
    # Nullable integer columns must not pass through float64: 64-bit bootstrap
    # seeds would otherwise lose identity-changing precision before Parquet.
    for column_name, column in contract.columns.items():
        if column.dtype == "integer" and column.nullable:
            frame[column_name] = pd.array(
                cast(Sequence[Any], frame[column_name].tolist()),
                dtype="Int64",
            )
    return _validate_table(name, frame)


def _replicate_table(campaign: G3FrequencyCalibrationCampaign) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in campaign.replicates:
        rows.append(
            {
                "replicate_id": item.replicate_id,
                "protocol_id": item.protocol_id,
                "scenario_id": campaign.protocol.scenario_id(
                    item.dependence_structure, item.non_null_prevalence
                ),
                "dependence_structure": G3FrequencyDependenceStructure(
                    item.dependence_structure
                ).value,
                "non_null_prevalence": item.non_null_prevalence,
                "replicate_index": item.replicate_index,
                "seed_lineage_json": canonical_json(item.seed_lineage.to_dict()),
                "source_artifact_id": item.source_artifact_id,
                "status": G3FrequencyReplicateStatus(item.status).value,
                "reason_code": item.reason_code,
                "permutation_null_p_value": item.permutation_null_p_value,
                "n_hypotheses": len(item.hypothesis_ids),
            }
        )
    return _empty_or_frame(G3F_CALIBRATION_REPLICATE_TABLE, rows)


def _hypothesis_table(campaign: G3FrequencyCalibrationCampaign) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in campaign.replicates:
        scenario_id = campaign.protocol.scenario_id(
            item.dependence_structure, item.non_null_prevalence
        )
        for index, hypothesis_id in enumerate(item.hypothesis_ids):
            candidate_width = float(item.ci_interval_width[index])
            baseline_width = float(item.baseline_ci_interval_width[index])
            candidate_value = None if math.isnan(candidate_width) else candidate_width
            baseline_value = None if math.isnan(baseline_width) else baseline_width
            values: dict[str, object] = {
                "replicate_id": item.replicate_id,
                "protocol_id": item.protocol_id,
                "scenario_id": scenario_id,
                "hypothesis_index": index,
                "hypothesis_id": hypothesis_id,
                "hypothesis_role": HypothesisRole(
                    item.hypothesis_roles[index]
                ).value,
                "parent_hypothesis_id": item.parent_hypothesis_ids[index],
                "truth_non_null": bool(item.truth_non_null[index]),
                "candidate_rejected_at_alpha": bool(item.rejected_at_alpha[index]),
                "baseline_rejected_at_alpha": bool(
                    item.baseline_rejected_at_alpha[index]
                ),
                "ci_contains_truth": bool(item.ci_contains_truth[index]),
                "candidate_ci_interval_width": candidate_value,
                "baseline_ci_interval_width": baseline_value,
            }
            values["hypothesis_row_id"] = _hypothesis_row_id(
                **cast(dict[str, Any], values)
            )
            rows.append(
                {
                    "hypothesis_row_id": values["hypothesis_row_id"],
                    **{
                        key: value
                        for key, value in values.items()
                        if key != "hypothesis_row_id"
                    },
                }
            )
    return _empty_or_frame(G3F_CALIBRATION_HYPOTHESIS_TABLE, rows)


def _scenario_table(campaign: G3FrequencyCalibrationCampaign) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in campaign.scenarios:
        rows.append(
            {
                "scenario_result_id": item.scenario_result_id,
                "scenario_id": item.scenario_id,
                "protocol_id": item.protocol_id,
                "dependence_structure": item.dependence_structure,
                "non_null_prevalence": item.non_null_prevalence,
                "metric": item.metric.value,
                "status": item.status.value,
                "reason_code": item.reason_code,
                "n_replicates": item.n_replicates,
                "n_observed_replicates": item.n_observed_replicates,
                "n_metric_replicates": item.n_metric_replicates,
                "replicate_ids_json": canonical_json(list(item.replicate_ids)),
                "point_estimate": item.point_estimate,
                "one_sided_bound": item.one_sided_bound,
                "bootstrap_seed": item.bootstrap_seed,
                "noninferiority_baseline_id": item.noninferiority_baseline_id,
                "noninferiority_estimand": item.noninferiority_estimand,
                "noninferiority_direction": item.noninferiority_direction,
                "noninferiority_margin": item.noninferiority_margin,
                "metric_weighting": _METRIC_WEIGHTING,
            }
        )
    return _empty_or_frame(G3F_CALIBRATION_SCENARIO_TABLE, rows)


def _permutation_table(campaign: G3FrequencyCalibrationCampaign) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "diagnostic_id": item.diagnostic_id,
            "protocol_id": item.protocol_id,
            "dependence_structure": item.dependence_structure,
            "status": item.status.value,
            "reason_code": item.reason_code,
            "n_replicates": item.n_replicates,
            "p_values_json": canonical_json([float(value) for value in item.p_values]),
            "ks_distance": item.ks_distance,
            "dkw_limit": item.dkw_limit,
        }
        for item in campaign.permutation_diagnostics
    ]
    return _empty_or_frame(G3F_CALIBRATION_PERMUTATION_TABLE, rows)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _mark_incomplete(path: Path, error_type: str | None = None) -> None:
    marker: dict[str, object] = {
        "artifact_schema_version": G3F_CALIBRATION_RESULT_SCHEMA_VERSION,
        "artifact_kind": G3F_CALIBRATION_ARTIFACT_KIND,
        "status": _INCOMPLETE,
    }
    if error_type is not None:
        marker["error_type"] = error_type
    try:
        _write_json(path / _STATUS_FILENAME, marker)
    except OSError:
        pass


def _table_record(
    path: Path,
    table: pd.DataFrame,
    contract: _TableContract,
) -> dict[str, object]:
    return {
        "filename": contract.filename,
        "rows": len(table),
        "sha256": _sha256_file(path),
        "schema": contract.schema_filename,
    }


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, object]:
    return {str(key): value for key, value in manifest.items() if key != "artifact_id"}


_MANIFEST_FIELDS = {
    "artifact_schema_version",
    "artifact_kind",
    "artifact_id",
    "status",
    "campaign_schema_version",
    "campaign_id",
    "protocol_id",
    "evidence_id",
    "gate_id",
    "gate_status",
    "formal_release_allowed",
    "hierarchical_q_release_allowed",
    "replicate_ids",
    "scenario_result_ids",
    "permutation_diagnostic_ids",
    "protocol",
    "hypothesis_universe",
    "hierarchical_procedure",
    "evidence",
    "calibration_gate",
    "generator_attestation",
    "tables",
}

_ATTESTATION_FIELDS = {
    "attestation_id",
    "campaign_kind",
    "protocol_id",
    "generator_id",
    "registry_id",
    "generator_config_digest",
    "generator_implementation_digest",
    "generator_code_version",
    "release_approved",
    "replicate_ids",
    "request_ids",
    "ledger_digest",
    "replayed_ledger_digest",
    "schema_version",
}


def _identifier_list(
    value: object,
    *,
    field_name: str,
    exact_length: int | None = None,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (exact_length is not None and len(value) != exact_length)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        _fail(
            "G3-F calibration registry is incomplete or duplicated",
            code="invalid_g3f_calibration_registry",
            field_name=field_name,
        )
    return cast(list[str], value)


def _positive_n_jobs(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("n_jobs must be an integer >= 1")
    return value


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    document = load_schema_document("g3f_calibration_result.schema.json")
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        _fail(
            "G3-F result schema has no properties",
            code="invalid_g3f_calibration_schema",
            field_name="properties",
        )
    expected = {
        name: _const(
            properties,
            name,
            schema="g3f_calibration_result.schema.json",
        )
        for name in (
            "artifact_schema_version",
            "artifact_kind",
            "status",
            "campaign_schema_version",
        )
    }
    if set(manifest) != _MANIFEST_FIELDS or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        _fail(
            "G3-F calibration manifest violates schema v1",
            code="invalid_g3f_calibration_manifest",
            field_name="manifest",
        )
    for field_name in (
        "artifact_id",
        "campaign_id",
        "protocol_id",
        "evidence_id",
        "gate_id",
    ):
        value = manifest[field_name]
        if not isinstance(value, str) or not value or value != value.strip():
            _fail(
                "G3-F calibration manifest has an invalid identifier",
                code="invalid_g3f_calibration_manifest",
                field_name=field_name,
            )
    gate_status = manifest["gate_status"]
    formal = manifest["formal_release_allowed"]
    hierarchical = manifest["hierarchical_q_release_allowed"]
    if gate_status not in {"passed", "failed", "not_estimable"} or not isinstance(
        formal, bool
    ) or not isinstance(hierarchical, bool):
        _fail(
            "G3-F calibration manifest has an invalid release decision",
            code="invalid_g3f_calibration_manifest",
            field_name="gate_status",
        )
    if formal is not (gate_status == "passed") or (hierarchical and not formal):
        _fail(
            "G3-F release flags differ from the producer-derived gate status",
            code="g3f_calibration_release_mismatch",
            field_name="formal_release_allowed,hierarchical_q_release_allowed",
        )
    replicate_ids = _identifier_list(
        manifest["replicate_ids"], field_name="replicate_ids"
    )
    _identifier_list(
        manifest["scenario_result_ids"],
        field_name="scenario_result_ids",
        exact_length=81,
    )
    _identifier_list(
        manifest["permutation_diagnostic_ids"],
        field_name="permutation_diagnostic_ids",
        exact_length=3,
    )
    for field_name in (
        "protocol",
        "hypothesis_universe",
        "hierarchical_procedure",
        "evidence",
        "calibration_gate",
    ):
        if not isinstance(manifest[field_name], Mapping):
            _fail(
                "G3-F calibration registry is not an object",
                code="invalid_g3f_calibration_registry",
                field_name=field_name,
            )
    protocol = cast(Mapping[str, Any], manifest["protocol"])
    evidence = cast(Mapping[str, Any], manifest["evidence"])
    gate = cast(Mapping[str, Any], manifest["calibration_gate"])
    attestation = manifest["generator_attestation"]
    if attestation is None:
        if (
            evidence.get("generator_attestation_id") is not None
            or gate.get("generator_attestation_id") is not None
            or evidence.get("generator_verified") is not False
            or gate.get("generator_verified") is not False
        ):
            _fail(
                "G3-F generator verification lacks a replay attestation",
                code="g3f_calibration_attestation_mismatch",
                field_name="generator_attestation",
            )
    elif not isinstance(attestation, Mapping) or set(attestation) != (
        _ATTESTATION_FIELDS
    ):
        _fail(
            "G3-F replay attestation has invalid fields",
            code="invalid_g3f_calibration_attestation",
            field_name="generator_attestation",
        )
    else:
        typed = cast(Mapping[str, Any], attestation)
        attestation_id = typed.get("attestation_id")
        release_approved = typed.get("release_approved")
        if (
            not isinstance(attestation_id, str)
            or not attestation_id
            or not isinstance(release_approved, bool)
            or typed.get("campaign_kind") != "g3_frequency"
            or typed.get("protocol_id") != manifest["protocol_id"]
            or typed.get("generator_id") != protocol.get("generator_id")
            or evidence.get("generator_attestation_id") != attestation_id
            or gate.get("generator_attestation_id") != attestation_id
            or evidence.get("generator_verified") is not release_approved
            or gate.get("generator_verified") is not release_approved
            or typed.get("ledger_digest") != typed.get("replayed_ledger_digest")
            or typed.get("replicate_ids") != replicate_ids
        ):
            _fail(
                "G3-F replay attestation differs from campaign lineage",
                code="g3f_calibration_attestation_mismatch",
                field_name="generator_attestation",
            )
    raw_tables = manifest["tables"]
    if not isinstance(raw_tables, Mapping) or set(raw_tables) != set(_SCHEMA_FILES):
        _fail(
            "G3-F calibration table registry is incomplete",
            code="invalid_g3f_calibration_registry",
            field_name="tables",
        )
    for name in _SCHEMA_FILES:
        raw = raw_tables[name]
        contract = _table_contract(name)
        if not isinstance(raw, Mapping) or set(raw) != {
            "filename",
            "rows",
            "sha256",
            "schema",
        }:
            _fail(
                "G3-F calibration table record is invalid",
                code="invalid_g3f_calibration_manifest",
                field_name=name,
            )
        digest = raw["sha256"]
        rows = raw["rows"]
        if (
            raw["filename"] != contract.filename
            or raw["schema"] != contract.schema_filename
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail(
                "G3-F calibration table metadata is invalid",
                code="invalid_g3f_calibration_manifest",
                field_name=name,
            )
    if manifest["artifact_id"] != stable_id(
        "g3f_calibration_result",
        _manifest_payload(manifest),
        schema_version="1",
    ):
        _fail(
            "G3-F calibration artifact identity is inconsistent",
            code="g3f_calibration_artifact_identity_mismatch",
            field_name="artifact_id",
        )
    return manifest


def _require_attestation_registry(
    campaign: G3FrequencyCalibrationCampaign,
    replay_registry: CalibrationReplayRegistry | None,
) -> None:
    attestation = campaign.generator_attestation
    if attestation is None:
        return
    if replay_registry is None:
        raise ResultValidationError(
            "Attested G3-F campaigns require their replay registry",
            code="g3f_calibration_attestation_unavailable",
            field="replay_registry",
            remediation="Pass the exact package-owned replay registry",
        )
    try:
        replay_registry._require_intact()
        registration = replay_registry.resolve(campaign.protocol.generator_id)
        manifest = registration.manifest
        valid = (
            attestation.registry_id == replay_registry.registry_id
            and attestation.generator_id == manifest.generator_id
            and attestation.generator_config_digest == manifest.config_digest
            and attestation.generator_implementation_digest
            == manifest.implementation_digest
            and attestation.generator_code_version == manifest.code_version
        )
    except (AttributeError, ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "G3-F replay registry cannot verify this campaign lineage",
            code="g3f_calibration_attestation_registry_mismatch",
            field="replay_registry",
            remediation="Use the exact package-owned registry used for attestation",
        ) from error
    if not valid:
        _fail(
            "G3-F replay registry differs from the campaign attestation",
            code="g3f_calibration_attestation_registry_mismatch",
            field_name="replay_registry",
            remediation="Use the exact package-owned registry used for attestation",
        )


def write_g3f_calibration_result(
    destination: str | Path,
    *,
    campaign: G3FrequencyCalibrationCampaign,
    replay_registry: CalibrationReplayRegistry | None = None,
    n_jobs: int = 1,
) -> G3FCalibrationResult:
    """Atomically persist one G3-F campaign from producer-owned ledgers."""

    if not isinstance(campaign, G3FrequencyCalibrationCampaign):
        raise TypeError(
            "campaign must be a producer-owned G3FrequencyCalibrationCampaign"
        )
    jobs = _positive_n_jobs(n_jobs)
    campaign._require_intact()
    _require_attestation_registry(campaign, replay_registry)
    if campaign.generator_attestation is not None:
        if replay_registry is None:
            raise ResultValidationError(
                "Attested G3-F campaign requires its replay registry",
                code="g3f_calibration_attestation_unavailable",
                field="replay_registry",
                remediation="Pass the exact package-owned replay registry",
            )
        try:
            replayed_campaign = (
                summarize_attested_g3_frequency_calibration_campaign(
                    campaign.protocol,
                    campaign.replicates,
                    registry=replay_registry,
                    n_jobs=jobs,
                )
            )
        except (ContractError, TypeError, ValueError) as error:
            raise ResultValidationError(
                "G3-F campaign could not be replayed before persistence",
                code="g3f_calibration_prewrite_replay_failed",
                field="replay_registry",
                remediation=(
                    "Use the exact approved registry and intact raw campaign ledgers"
                ),
            ) from error
        if (
            replayed_campaign.campaign_id != campaign.campaign_id
            or replayed_campaign.evidence.to_dict() != campaign.evidence.to_dict()
            or replayed_campaign.generator_attestation is None
            or replayed_campaign.generator_attestation.to_dict()
            != campaign.generator_attestation.to_dict()
        ):
            raise ResultValidationError(
                "Prewrite G3-F replay differs from the supplied campaign",
                code="g3f_calibration_prewrite_replay_mismatch",
                field="campaign_id,generator_attestation",
                remediation="Regenerate the campaign with the exact approved registry",
            )
        campaign = replayed_campaign
    gate = build_g3_frequency_calibration_gate(campaign.evidence)
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"G3-F calibration destination {output.name!r} already exists",
            code="g3f_calibration_destination_exists",
            field="destination",
            remediation="Choose a new versioned result directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        tables = {
            G3F_CALIBRATION_REPLICATE_TABLE: _replicate_table(campaign),
            G3F_CALIBRATION_HYPOTHESIS_TABLE: _hypothesis_table(campaign),
            G3F_CALIBRATION_SCENARIO_TABLE: _scenario_table(campaign),
            G3F_CALIBRATION_PERMUTATION_TABLE: _permutation_table(campaign),
        }
        table_records: dict[str, dict[str, object]] = {}
        for name, table in tables.items():
            contract = _table_contract(name)
            path = temporary / contract.filename
            table.to_parquet(
                path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                row_group_size=131_072,
            )
            table_records[name] = _table_record(path, table, contract)
        manifest: dict[str, object] = {
            "artifact_schema_version": G3F_CALIBRATION_RESULT_SCHEMA_VERSION,
            "artifact_kind": G3F_CALIBRATION_ARTIFACT_KIND,
            "status": _COMPLETE,
            "campaign_schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": campaign.campaign_id,
            "protocol_id": campaign.protocol.protocol_id,
            "evidence_id": campaign.evidence.evidence_id,
            "gate_id": gate.gate_id,
            "gate_status": gate.status.value,
            "formal_release_allowed": gate.formal_release_allowed,
            "hierarchical_q_release_allowed": gate.hierarchical_q_release_allowed,
            "replicate_ids": [item.replicate_id for item in campaign.replicates],
            "scenario_result_ids": [
                item.scenario_result_id for item in campaign.scenarios
            ],
            "permutation_diagnostic_ids": [
                item.diagnostic_id for item in campaign.permutation_diagnostics
            ],
            "protocol": campaign.protocol.to_dict(),
            "hypothesis_universe": campaign.protocol.hypothesis_universe.to_dict(),
            "hierarchical_procedure": (
                campaign.protocol.hierarchical_procedure.to_dict()
            ),
            "evidence": campaign.evidence.to_dict(),
            "calibration_gate": gate.to_dict(),
            "generator_attestation": (
                None
                if campaign.generator_attestation is None
                else campaign.generator_attestation.to_dict()
            ),
            "tables": table_records,
        }
        manifest["artifact_id"] = stable_id(
            "g3f_calibration_result", manifest, schema_version="1"
        )
        _validate_manifest(cast(dict[str, Any], manifest))
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "artifact_schema_version": G3F_CALIBRATION_RESULT_SCHEMA_VERSION,
                "artifact_kind": G3F_CALIBRATION_ARTIFACT_KIND,
                "status": _COMPLETE,
                "artifact_id": manifest["artifact_id"],
            },
        )
        os.replace(temporary, output)
    except Exception as error:
        if temporary.exists():
            _mark_incomplete(temporary, type(error).__name__)
            if not output.exists():
                os.replace(temporary, output)
        raise ResultWriteError(
            "G3-F calibration result write failed and was marked incomplete",
            code="g3f_calibration_write_failed",
            field="destination",
            remediation="Inspect producer diagnostics and write to a new directory",
        ) from error
    return G3FCalibrationResult.load(
        output,
        replay_registry=replay_registry,
        n_jobs=jobs,
    )


def _load_table(
    root: Path,
    name: str,
    record: Mapping[str, Any],
) -> pd.DataFrame:
    contract = _table_contract(name)
    path = root / contract.filename
    try:
        digest = _sha256_file(path)
    except Exception as error:
        raise ResultValidationError(
            f"G3-F calibration table {name!r} is missing or corrupted",
            code="corrupted_g3f_calibration_table",
            field=name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if digest != record["sha256"]:
        _fail(
            f"G3-F calibration table {name!r} does not match its manifest",
            code="g3f_calibration_digest_mismatch",
            field_name=name,
        )
    try:
        frame = _read_parquet(path, engine="pyarrow")
    except Exception as error:
        raise ResultValidationError(
            f"G3-F calibration table {name!r} is missing or corrupted",
            code="corrupted_g3f_calibration_table",
            field=name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if len(frame) != record["rows"]:
        _fail(
            f"G3-F calibration table {name!r} row count differs",
            code="g3f_calibration_digest_mismatch",
            field_name=name,
        )
    return _validate_table(name, frame)


def _universe_from_manifest(manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
    raw = manifest["hypothesis_universe"]
    if not isinstance(raw, Mapping) or not isinstance(raw.get("declarations"), list):
        _fail(
            "G3-F hypothesis universe registry is invalid",
            code="invalid_g3f_calibration_registry",
            field_name="hypothesis_universe",
        )
    declarations: list[HypothesisDeclaration] = []
    try:
        for value in cast(list[object], raw["declarations"]):
            if not isinstance(value, Mapping):
                raise TypeError("declaration must be an object")
            item = cast(Mapping[str, Any], value)
            declaration = HypothesisDeclaration(
                endpoint=item["endpoint"],
                contrast_name=item["contrast_name"],
                receiver=item["receiver"],
                family_id=item["family_id"],
                mode=item["mode"],
                role=item["role"],
                multiplicity_family=item["multiplicity_family"],
                parent_key=item["parent_key"],
                filter_stage=item["filter_stage"],
                prefilter_policy=item["prefilter_policy"],
                prefilter_status=item["prefilter_status"],
                filter_reason_code=item["filter_reason_code"],
            )
            if declaration.to_dict() != dict(item):
                raise ValueError("declaration identity mismatch")
            declarations.append(declaration)
        universe = freeze_hypothesis_universe(
            declarations,
            universe_name=cast(str, raw["universe_name"]),
        )
    except (KeyError, ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted G3-F hypothesis universe cannot be reconstructed",
            code="invalid_g3f_calibration_universe",
            field="hypothesis_universe",
            remediation="Reject the artifact and regenerate it",
        ) from error
    if universe.to_dict() != dict(raw):
        _fail(
            "Persisted G3-F hypothesis universe identity differs",
            code="g3f_calibration_universe_mismatch",
            field_name="hypothesis_universe",
        )
    return universe


def _procedure_from_manifest(manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
    raw = manifest["hierarchical_procedure"]
    procedure = freeze_hierarchical_fdr_spec()
    if not isinstance(raw, Mapping) or procedure.to_dict() != dict(raw):
        _fail(
            "Persisted G3-F hierarchical procedure identity differs",
            code="g3f_calibration_procedure_mismatch",
            field_name="hierarchical_procedure",
        )
    return procedure


def _protocol_from_manifest(
    manifest: Mapping[str, Any],
) -> G3FrequencyCalibrationProtocol:
    raw = manifest["protocol"]
    if not isinstance(raw, Mapping):
        _fail(
            "G3-F protocol registry is invalid",
            code="invalid_g3f_calibration_registry",
            field_name="protocol",
        )
    seed_value = raw.get("seed_lineage")
    generator_value = raw.get("generator_manifest")
    if not isinstance(seed_value, Mapping) or not isinstance(generator_value, Mapping):
        _fail(
            "G3-F protocol lineage registry is invalid",
            code="invalid_g3f_calibration_registry",
            field_name="protocol.seed_lineage,generator_manifest",
        )
    try:
        generator = CalibrationGeneratorManifest.from_dict(
            cast(Mapping[str, object], generator_value)
        )
        protocol = G3FrequencyCalibrationProtocol(
            campaign_name=cast(str, raw["campaign_name"]),
            generator_manifest=generator,
            noninferiority_baseline_id=cast(str, raw["noninferiority_baseline_id"]),
            hypothesis_universe=_universe_from_manifest(manifest),
            hierarchical_procedure=_procedure_from_manifest(manifest),
            seed_lineage=SeedLineage.from_dict(cast(Mapping[str, object], seed_value)),
            schema_version=cast(str, raw["schema_version"]),
        )
    except (KeyError, ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted G3-F protocol cannot be reconstructed",
            code="invalid_g3f_calibration_protocol",
            field="protocol",
            remediation="Reject the artifact and regenerate it",
        ) from error
    if (
        protocol.to_dict() != dict(raw)
        or raw.get("generator_id") != generator.generator_id
        or protocol.protocol_id != manifest["protocol_id"]
    ):
        _fail(
            "Persisted G3-F protocol identity differs",
            code="g3f_calibration_protocol_mismatch",
            field_name="protocol_id",
        )
    return protocol


def _parse_seed_lineage(value: object) -> SeedLineage:
    if not isinstance(value, str):
        _fail(
            "G3-F replicate seed lineage is not JSON",
            code="invalid_g3f_calibration_seed_lineage",
            field_name="seed_lineage_json",
        )
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping) or canonical_json(parsed) != value:
            raise ValueError("seed lineage JSON is not canonical")
        return SeedLineage.from_dict(cast(Mapping[str, object], parsed))
    except (ContractError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ResultValidationError(
            "Persisted G3-F replicate seed lineage is invalid",
            code="invalid_g3f_calibration_seed_lineage",
            field="seed_lineage_json",
            remediation="Reject the artifact and regenerate it",
        ) from error


def _nullable_float(value: object) -> float | None:
    return None if _is_missing(value) else float(cast(numbers.Real, value))


def _replicates_from_tables(
    protocol: G3FrequencyCalibrationProtocol,
    replicate_frame: pd.DataFrame,
    hypothesis_frame: pd.DataFrame,
) -> tuple[G3FrequencyCalibrationReplicate, ...]:
    replicate_ids = set(replicate_frame["replicate_id"].astype(str))
    hypothesis_replicate_ids = set(hypothesis_frame["replicate_id"].astype(str))
    if hypothesis_replicate_ids != replicate_ids:
        _fail(
            "G3-F hypothesis rows do not cover the exact replicate registry",
            code="g3f_calibration_hypothesis_coverage_mismatch",
            field_name="replicate_id",
        )
    for frame in (replicate_frame, hypothesis_frame):
        if set(frame["protocol_id"].astype(str)) not in (set(), {protocol.protocol_id}):
            _fail(
                "G3-F ledger protocol links differ from the manifest",
                code="g3f_calibration_protocol_link_mismatch",
                field_name="protocol_id",
            )
    reconstructed: list[G3FrequencyCalibrationReplicate] = []
    for row in replicate_frame.itertuples(index=False):
        source = cast(Any, row)
        replicate_id = str(source.replicate_id)
        hypotheses = hypothesis_frame.loc[
            hypothesis_frame["replicate_id"].astype(str) == replicate_id
        ].sort_values("hypothesis_index", kind="stable")
        n_hypotheses = int(source.n_hypotheses)
        if (
            tuple(int(value) for value in hypotheses["hypothesis_index"])
            != tuple(range(n_hypotheses))
            or len(hypotheses) != n_hypotheses
        ):
            _fail(
                "G3-F hypothesis indexes do not form a complete replicate ledger",
                code="g3f_calibration_hypothesis_coverage_mismatch",
                field_name="hypothesis_index",
            )
        prevalence = _nullable_float(source.non_null_prevalence)
        expected_scenario = protocol.scenario_id(
            str(source.dependence_structure), prevalence
        )
        if str(source.scenario_id) != expected_scenario or set(
            hypotheses["scenario_id"].astype(str)
        ) != {expected_scenario}:
            _fail(
                "G3-F replicate scenario links differ from the protocol",
                code="g3f_calibration_scenario_link_mismatch",
                field_name="scenario_id",
            )
        candidate_width = np.asarray(
            [
                np.nan if _is_missing(value) else float(value)
                for value in hypotheses["candidate_ci_interval_width"].array
            ],
            dtype="<f8",
        )
        baseline_width = np.asarray(
            [
                np.nan if _is_missing(value) else float(value)
                for value in hypotheses["baseline_ci_interval_width"].array
            ],
            dtype="<f8",
        )
        try:
            replicate = G3FrequencyCalibrationReplicate(
                protocol_id=str(source.protocol_id),
                dependence_structure=str(source.dependence_structure),
                non_null_prevalence=prevalence,
                replicate_index=int(source.replicate_index),
                seed_lineage=_parse_seed_lineage(source.seed_lineage_json),
                source_artifact_id=str(source.source_artifact_id),
                hypothesis_ids=tuple(hypotheses["hypothesis_id"].astype(str)),
                hypothesis_roles=tuple(hypotheses["hypothesis_role"].astype(str)),
                parent_hypothesis_ids=tuple(
                    None if _is_missing(value) else str(value)
                    for value in hypotheses["parent_hypothesis_id"].array
                ),
                truth_non_null=np.asarray(hypotheses["truth_non_null"], dtype="|b1"),
                rejected_at_alpha=np.asarray(
                    hypotheses["candidate_rejected_at_alpha"], dtype="|b1"
                ),
                baseline_rejected_at_alpha=np.asarray(
                    hypotheses["baseline_rejected_at_alpha"], dtype="|b1"
                ),
                ci_contains_truth=np.asarray(
                    hypotheses["ci_contains_truth"], dtype="|b1"
                ),
                ci_interval_width=candidate_width,
                baseline_ci_interval_width=baseline_width,
                permutation_null_p_value=_nullable_float(
                    source.permutation_null_p_value
                ),
                status=str(source.status),
                reason_code=(
                    None if _is_missing(source.reason_code) else str(source.reason_code)
                ),
            )
        except (ContractError, TypeError, ValueError) as error:
            raise ResultValidationError(
                "Persisted G3-F replicate cannot be reconstructed",
                code="invalid_g3f_calibration_replicate",
                field="replicate_id",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if replicate.replicate_id != replicate_id:
            _fail(
                "Persisted G3-F replicate identity cannot be reconstructed",
                code="g3f_calibration_replicate_identity_mismatch",
                field_name="replicate_id",
            )
        reconstructed.append(replicate)
    return tuple(reconstructed)


def _canonical_frame_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for row in frame.itertuples(index=False, name=None):
        values: dict[str, object] = {}
        for column_name, value in zip(frame.columns, row, strict=True):
            if _is_missing(value):
                normalized: object = None
            elif isinstance(value, np.generic):
                normalized = cast(Any, value).item()
            else:
                normalized = cast(object, value)
            values[str(column_name)] = normalized
        payload.append(values)
    return payload


def _parse_canonical_json_array(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, str):
        _fail(
            "G3-F derived registry is not canonical JSON",
            code="invalid_g3f_calibration_derived_registry",
            field_name=field_name,
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ResultValidationError(
            "G3-F derived registry contains invalid JSON",
            code="invalid_g3f_calibration_derived_registry",
            field=field_name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if not isinstance(parsed, list) or canonical_json(parsed) != value:
        _fail(
            "G3-F derived registry is not a canonical JSON array",
            code="invalid_g3f_calibration_derived_registry",
            field_name=field_name,
        )
    return parsed


def _validate_derived_json(tables: Mapping[str, pd.DataFrame]) -> None:
    for value in tables[G3F_CALIBRATION_SCENARIO_TABLE]["replicate_ids_json"]:
        parsed = _parse_canonical_json_array(value, field_name="replicate_ids_json")
        if any(not isinstance(item, str) or not item for item in parsed) or len(
            parsed
        ) != len(set(parsed)):
            _fail(
                "G3-F scenario replicate registry is malformed or duplicated",
                code="invalid_g3f_calibration_derived_registry",
                field_name="replicate_ids_json",
            )
    for value in tables[G3F_CALIBRATION_PERMUTATION_TABLE]["p_values_json"]:
        parsed = _parse_canonical_json_array(value, field_name="p_values_json")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in parsed
        ):
            _fail(
                "G3-F permutation p-value registry is malformed",
                code="invalid_g3f_calibration_derived_registry",
                field_name="p_values_json",
            )


def _require_exact_table(
    name: str,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    if canonical_json(_canonical_frame_payload(observed)) != canonical_json(
        _canonical_frame_payload(expected)
    ):
        _fail(
            f"Persisted G3-F table {name!r} differs from producer reconstruction",
            code="g3f_calibration_reconstruction_mismatch",
            field_name=name,
        )


def _validate_links(
    manifest: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    *,
    replay_registry: CalibrationReplayRegistry | None,
    n_jobs: int,
) -> tuple[G3FrequencyCalibrationCampaign, G3FrequencyCalibrationGate]:
    _validate_derived_json(tables)
    protocol = _protocol_from_manifest(manifest)
    replicates = _replicates_from_tables(
        protocol,
        tables[G3F_CALIBRATION_REPLICATE_TABLE],
        tables[G3F_CALIBRATION_HYPOTHESIS_TABLE],
    )
    if manifest["generator_attestation"] is not None and replay_registry is None:
        raise ResultValidationError(
            "Attested G3-F artifact requires its replay registry",
            code="g3f_calibration_attestation_unavailable",
            field="replay_registry",
            remediation="Load with the exact package-owned replay registry",
        )
    try:
        if manifest["generator_attestation"] is None:
            campaign = summarize_g3_frequency_calibration_campaign(
                protocol, replicates
            )
        else:
            if replay_registry is None:
                raise ResultValidationError(
                    "Attested G3-F artifact requires its replay registry",
                    code="g3f_calibration_attestation_unavailable",
                    field="replay_registry",
                    remediation="Load with the exact package-owned replay registry",
                )
            campaign = summarize_attested_g3_frequency_calibration_campaign(
                protocol,
                replicates,
                registry=replay_registry,
                n_jobs=n_jobs,
            )
        gate = build_g3_frequency_calibration_gate(campaign.evidence)
    except (ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted G3-F ledgers do not form a valid campaign",
            code="invalid_g3f_calibration_campaign",
            field="replicates,hypotheses",
            remediation="Reject the artifact and regenerate it",
        ) from error
    expected_tables = {
        G3F_CALIBRATION_REPLICATE_TABLE: _replicate_table(campaign),
        G3F_CALIBRATION_HYPOTHESIS_TABLE: _hypothesis_table(campaign),
        G3F_CALIBRATION_SCENARIO_TABLE: _scenario_table(campaign),
        G3F_CALIBRATION_PERMUTATION_TABLE: _permutation_table(campaign),
    }
    for name, expected in expected_tables.items():
        _require_exact_table(name, tables[name], expected)
    if (
        manifest["campaign_id"] != campaign.campaign_id
        or manifest["protocol_id"] != campaign.protocol.protocol_id
        or manifest["evidence_id"] != campaign.evidence.evidence_id
        or manifest["gate_id"] != gate.gate_id
        or manifest["gate_status"] != gate.status.value
        or manifest["formal_release_allowed"] != gate.formal_release_allowed
        or manifest["hierarchical_q_release_allowed"]
        != gate.hierarchical_q_release_allowed
        or manifest["replicate_ids"]
        != [item.replicate_id for item in campaign.replicates]
        or manifest["scenario_result_ids"]
        != [item.scenario_result_id for item in campaign.scenarios]
        or manifest["permutation_diagnostic_ids"]
        != [item.diagnostic_id for item in campaign.permutation_diagnostics]
        or manifest["protocol"] != campaign.protocol.to_dict()
        or manifest["hypothesis_universe"]
        != campaign.protocol.hypothesis_universe.to_dict()
        or manifest["hierarchical_procedure"]
        != campaign.protocol.hierarchical_procedure.to_dict()
        or manifest["evidence"] != campaign.evidence.to_dict()
        or manifest["calibration_gate"] != gate.to_dict()
        or manifest["generator_attestation"]
        != (
            None
            if campaign.generator_attestation is None
            else campaign.generator_attestation.to_dict()
        )
    ):
        _fail(
            "G3-F campaign, evidence, diagnostics, or gate differs from reconstruction",
            code="g3f_calibration_linkage_mismatch",
            field_name="campaign_id,evidence_id,gate_id",
        )
    return campaign, gate


@dataclass(frozen=True, slots=True, init=False)
class G3FCalibrationResult:
    """Immutable facade over one independent G3-F campaign directory."""

    path: Path
    _manifest: dict[str, Any] = field(repr=False)
    _calibration_gate: G3FrequencyCalibrationGate = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "G3FCalibrationResult is producer-owned; use "
            "write_g3f_calibration_result() or load()"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        replay_registry: CalibrationReplayRegistry | None = None,
        n_jobs: int = 1,
    ) -> G3FCalibrationResult:
        """Load a completed artifact, replaying attested campaigns exactly."""

        jobs = _positive_n_jobs(n_jobs)
        root = Path(path)
        try:
            marker = _read_json(root / _STATUS_FILENAME)
        except Exception as error:
            raise ResultValidationError(
                "G3-F calibration status marker is missing or corrupted",
                code="invalid_g3f_calibration_status",
                field="path",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if marker.get("status") != _COMPLETE:
            raise IncompleteResultError(
                "G3-F calibration result is incomplete",
                code="incomplete_g3f_calibration_result",
                field="path",
                remediation="Inspect producer diagnostics and rerun",
            )
        if set(marker) != {
            "artifact_schema_version",
            "artifact_kind",
            "status",
            "artifact_id",
        } or (
            marker.get("artifact_schema_version")
            != G3F_CALIBRATION_RESULT_SCHEMA_VERSION
            or marker.get("artifact_kind") != G3F_CALIBRATION_ARTIFACT_KIND
        ):
            _fail(
                "G3-F calibration status marker violates schema v1",
                code="invalid_g3f_calibration_status",
                field_name="path",
            )
        try:
            manifest = _validate_manifest(_read_json(root / _MANIFEST_FILENAME))
        except ResultValidationError:
            raise
        except Exception as error:
            raise ResultValidationError(
                "G3-F calibration manifest is missing or corrupted",
                code="invalid_g3f_calibration_manifest",
                field="path",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if marker["artifact_id"] != manifest["artifact_id"]:
            _fail(
                "G3-F calibration status and manifest identities differ",
                code="g3f_calibration_status_manifest_mismatch",
                field_name="artifact_id",
            )
        raw_records = cast(dict[str, Mapping[str, Any]], manifest["tables"])
        tables = {
            name: _load_table(root, name, record)
            for name, record in raw_records.items()
        }
        _, gate = _validate_links(
            manifest,
            tables,
            replay_registry=replay_registry,
            n_jobs=jobs,
        )
        self = object.__new__(cls)
        object.__setattr__(self, "path", root.resolve())
        object.__setattr__(self, "_manifest", copy.deepcopy(manifest))
        object.__setattr__(self, "_calibration_gate", gate)
        return self

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def calibration_gate(self) -> G3FrequencyCalibrationGate:
        return self._calibration_gate

    def _read(
        self,
        name: str,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        contract = _table_contract(name)
        requested = None if columns is None else list(columns)
        if requested is not None:
            unknown = set(requested).difference(contract.columns)
            if unknown:
                raise KeyError(sorted(unknown)[0])
        parquet_filters: list[tuple[str, str, object]] | None = None
        if filters:
            unknown = set(filters).difference(contract.columns)
            if unknown:
                raise KeyError(sorted(unknown)[0])
            parquet_filters = [(key, "==", value) for key, value in filters.items()]
        return _read_parquet(
            self.path / contract.filename,
            columns=requested,
            filters=parquet_filters,
            engine="pyarrow",
        )

    def read_replicates(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self._read(
            G3F_CALIBRATION_REPLICATE_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_hypotheses(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self._read(
            G3F_CALIBRATION_HYPOTHESIS_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_scenarios(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self._read(
            G3F_CALIBRATION_SCENARIO_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_permutation_diagnostics(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self._read(
            G3F_CALIBRATION_PERMUTATION_TABLE,
            filters=filters,
            columns=columns,
        )


__all__ = [
    "G3F_CALIBRATION_ARTIFACT_KIND",
    "G3F_CALIBRATION_HYPOTHESIS_TABLE",
    "G3F_CALIBRATION_PERMUTATION_TABLE",
    "G3F_CALIBRATION_REPLICATE_TABLE",
    "G3F_CALIBRATION_RESULT_SCHEMA_VERSION",
    "G3F_CALIBRATION_SCENARIO_TABLE",
    "G3FCalibrationResult",
    "write_g3f_calibration_result",
]
