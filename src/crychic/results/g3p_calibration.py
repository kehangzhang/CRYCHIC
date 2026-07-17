"""Independent, integrity-checked persistence for ADR-013 G3-P campaigns.

The raw replicate and candidate ledgers are authoritative.  Loading an
artifact reconstructs the producer-owned campaign from those ledgers and
re-derives scenario metrics, evidence, and the release gate.  Persisted
summary values therefore cannot authorize probability release on their own.
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
from crychic.inference import (
    G3PCalibrationCampaign,
    G3PCalibrationGate,
    G3PCalibrationProtocol,
    G3PCandidateStatus,
    G3PDependenceStructure,
    G3PReplicatePredictions,
    G3PReplicateStatus,
    build_g3p_calibration_gate,
    summarize_attested_g3p_calibration_campaign,
    summarize_g3p_calibration_campaign,
)
from crychic.inference.calibration_attestation import (
    CalibrationGeneratorManifest,
    CalibrationReplayRegistry,
)

from ._schema import load_schema_document
from .errors import IncompleteResultError, ResultValidationError, ResultWriteError

G3P_CALIBRATION_RESULT_SCHEMA_VERSION = "2.0.0"
G3P_CALIBRATION_ARTIFACT_KIND = "adr013_g3p_calibration_result"

G3P_CALIBRATION_REPLICATE_TABLE = "g3p_calibration_replicates"
G3P_CALIBRATION_CANDIDATE_TABLE = "g3p_calibration_candidates"
G3P_CALIBRATION_SCENARIO_TABLE = "g3p_calibration_scenarios"

_CAMPAIGN_SCHEMA_VERSION = "3.0.0"
_TABLE_SCHEMA_VERSION = "1.0.0"
_STATUS_FILENAME = "_status.json"
_MANIFEST_FILENAME = "g3p_calibration_manifest.json"
_COMPLETE = "complete"
_INCOMPLETE = "incomplete"
_METRIC_WEIGHTING = "candidate_equal_within_replicate_then_replicate_equal_v1"

_SCHEMA_FILES = {
    G3P_CALIBRATION_REPLICATE_TABLE: "g3p_calibration_replicates.schema.json",
    G3P_CALIBRATION_CANDIDATE_TABLE: "g3p_calibration_candidates.schema.json",
    G3P_CALIBRATION_SCENARIO_TABLE: "g3p_calibration_scenarios.schema.json",
}


def _read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
    """Call pandas without version-specific PyArrow passthrough keywords."""

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
            f"G3-P schema {schema!r} lacks {name!r}",
            code="invalid_g3p_calibration_schema",
            field_name=name,
            remediation="Restore the released G3-P calibration schemas",
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
            f"G3-P schema {schema_filename!r} has no properties",
            code="invalid_g3p_calibration_schema",
            field_name="properties",
            remediation="Restore the released G3-P calibration schemas",
        )
    version = _const(properties, "artifact_schema_version", schema=schema_filename)
    table_name = _const(properties, "table", schema=schema_filename)
    filename = _const(properties, "filename", schema=schema_filename)
    primary_key = _const(properties, "primary_key", schema=schema_filename)
    logical_key = _const(properties, "logical_key", schema=schema_filename)
    _const(properties, "foreign_keys", schema=schema_filename)
    columns_block = properties.get("columns")
    raw_columns = (
        columns_block.get("properties") if isinstance(columns_block, Mapping) else None
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
            f"G3-P schema {schema_filename!r} is incompatible",
            code="invalid_g3p_calibration_schema",
            field_name="artifact_schema_version",
            remediation="Use schemas from this artifact implementation",
        )
    columns: dict[str, _ColumnContract] = {}
    for column_name, raw in raw_columns.items():
        if not isinstance(column_name, str) or not isinstance(raw, Mapping):
            _fail(
                "G3-P table has an invalid column contract",
                code="invalid_g3p_calibration_schema",
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
                f"G3-P column {column_name!r} has an invalid contract",
                code="invalid_g3p_calibration_schema",
                field_name=column_name,
            )
        columns[column_name] = _ColumnContract(
            dtype=dtype,
            nullable=nullable,
            enum=tuple(enum),
            minimum=None if minimum is None else float(minimum),
            maximum=None if maximum is None else float(maximum),
        )
    keys = tuple(cast(list[str], primary_key)) + tuple(cast(list[str], logical_key))
    if not set(keys).issubset(columns):
        _fail(
            "G3-P table key is absent from its columns",
            code="invalid_g3p_calibration_schema",
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
                code="invalid_g3p_calibration_null",
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
            code="invalid_g3p_calibration_dtype",
            field_name=column_name,
        )
    if contract.enum and value not in contract.enum:
        _fail(
            f"{table_name}.{column_name} is outside its enum",
            code="invalid_g3p_calibration_enum",
            field_name=column_name,
        )
    if contract.dtype in {"integer", "float"}:
        numeric = float(cast(numbers.Real, value))
        if contract.minimum is not None and numeric < contract.minimum:
            _fail(
                f"{table_name}.{column_name} is below its minimum",
                code="invalid_g3p_calibration_range",
                field_name=column_name,
            )
        if contract.maximum is not None and numeric > contract.maximum:
            _fail(
                f"{table_name}.{column_name} exceeds its maximum",
                code="invalid_g3p_calibration_range",
                field_name=column_name,
            )


def _validate_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    contract = _table_contract(name)
    expected_columns = tuple(contract.columns)
    if tuple(frame.columns) != expected_columns:
        _fail(
            f"G3-P table {name!r} columns do not match its schema",
            code="invalid_g3p_calibration_columns",
            field_name=name,
        )
    if frame.duplicated(list(contract.primary_key)).any():
        _fail(
            f"G3-P table {name!r} has duplicate primary keys",
            code="duplicate_g3p_calibration_primary_key",
            field_name=",".join(contract.primary_key),
        )
    if frame.duplicated(list(contract.logical_key)).any():
        _fail(
            f"G3-P table {name!r} has duplicate logical keys",
            code="duplicate_g3p_calibration_logical_key",
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


def _candidate_row_id(
    *,
    protocol_id: str,
    scenario_id: str,
    replicate_id: str,
    candidate_index: int,
    candidate_edge_id: str,
    stratum_id: str,
    candidate_status: str,
    candidate_reason_code: str | None,
    truth_active: bool,
    candidate_probability: float | None,
) -> str:
    identifier: str = stable_id(
        "g3p_calibration_candidate_row",
        {
            "protocol_id": protocol_id,
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "candidate_index": candidate_index,
            "candidate_edge_id": candidate_edge_id,
            "stratum_id": stratum_id,
            "candidate_status": candidate_status,
            "candidate_reason_code": candidate_reason_code,
            "truth_active": truth_active,
            "candidate_probability": candidate_probability,
        },
        schema_version="1",
    )
    return identifier


def _replicate_table(
    campaign: G3PCalibrationCampaign,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for replicate in campaign.replicates:
        scenario_id = campaign.protocol.scenario_id(
            replicate.dependence_structure,
            replicate.non_null_prevalence,
        )
        rows.append(
            {
                "replicate_id": replicate.replicate_id,
                "protocol_id": replicate.protocol_id,
                "scenario_id": scenario_id,
                "dependence_structure": G3PDependenceStructure(
                    replicate.dependence_structure
                ).value,
                "non_null_prevalence": replicate.non_null_prevalence,
                "replicate_index": replicate.replicate_index,
                "seed_lineage_json": canonical_json(replicate.seed_lineage.to_dict()),
                "source_active_null_id": replicate.source_active_null_id,
                "source_candidate_universe_id": (
                    replicate.source_candidate_universe_id
                ),
                "source_probability_collection_id": (
                    replicate.source_probability_collection_id
                ),
                "status": G3PReplicateStatus(replicate.status).value,
                "reason_code": replicate.reason_code,
                "n_candidates": replicate.n_candidates,
                "n_eligible_candidates": replicate.n_eligible_candidates,
                "n_structural_zero_candidates": (
                    replicate.n_structural_zero_candidates
                ),
            }
        )
    frame = pd.DataFrame(
        rows, columns=tuple(_table_contract(G3P_CALIBRATION_REPLICATE_TABLE).columns)
    )
    return _validate_table(G3P_CALIBRATION_REPLICATE_TABLE, frame)


def _candidate_table(
    campaign: G3PCalibrationCampaign,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for replicate in campaign.replicates:
        scenario_id = campaign.protocol.scenario_id(
            replicate.dependence_structure,
            replicate.non_null_prevalence,
        )
        for index, candidate_edge_id in enumerate(replicate.candidate_edge_ids):
            raw_probability = float(replicate.candidate_probabilities[index])
            probability = None if math.isnan(raw_probability) else raw_probability
            truth_active = bool(replicate.truth_active[index])
            stratum_id = replicate.stratum_ids[index]
            candidate_status = G3PCandidateStatus(
                replicate.candidate_statuses[index]
            ).value
            candidate_reason = replicate.candidate_reason_codes[index]
            row_id = _candidate_row_id(
                protocol_id=replicate.protocol_id,
                scenario_id=scenario_id,
                replicate_id=replicate.replicate_id,
                candidate_index=index,
                candidate_edge_id=candidate_edge_id,
                stratum_id=stratum_id,
                candidate_status=candidate_status,
                candidate_reason_code=candidate_reason,
                truth_active=truth_active,
                candidate_probability=probability,
            )
            rows.append(
                {
                    "candidate_row_id": row_id,
                    "replicate_id": replicate.replicate_id,
                    "protocol_id": replicate.protocol_id,
                    "scenario_id": scenario_id,
                    "candidate_index": index,
                    "candidate_edge_id": candidate_edge_id,
                    "stratum_id": stratum_id,
                    "candidate_status": candidate_status,
                    "candidate_reason_code": candidate_reason,
                    "truth_active": truth_active,
                    "candidate_probability": probability,
                }
            )
    frame = pd.DataFrame(
        rows, columns=tuple(_table_contract(G3P_CALIBRATION_CANDIDATE_TABLE).columns)
    )
    return _validate_table(G3P_CALIBRATION_CANDIDATE_TABLE, frame)


def _scenario_table(
    campaign: G3PCalibrationCampaign,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in campaign.scenarios:
        value = scenario.to_dict()
        rows.append(
            {
                "scenario_result_id": value["scenario_result_id"],
                "scenario_id": value["scenario_id"],
                "protocol_id": value["protocol_id"],
                "dependence_structure": value["dependence_structure"],
                "non_null_prevalence": value["non_null_prevalence"],
                "status": value["status"],
                "reason_code": value["reason_code"],
                "n_replicates": value["n_replicates"],
                "n_observed_replicates": value["n_observed_replicates"],
                "minimum_eligible_candidates_per_stratum": value[
                    "minimum_eligible_candidates_per_stratum"
                ],
                "replicate_ids_json": canonical_json(value["replicate_ids"]),
                "stratum_result_ids_json": canonical_json(value["stratum_result_ids"]),
                "strata_json": canonical_json(value["strata"]),
                "observed_prevalence": value["observed_prevalence"],
                "brier_score": value["brier_score"],
                "prevalence_only_brier_score": value["prevalence_only_brier_score"],
                "brier_relative_improvement": value["brier_relative_improvement"],
                "ece": value["ece"],
                "ece_upper_bound": value["ece_upper_bound"],
                "calibration_in_the_large": value["calibration_in_the_large"],
                "calibration_slope": value["calibration_slope"],
                "bootstrap_seed": value["bootstrap_seed"],
                "metric_weighting": value["metric_weighting"],
            }
        )
    frame = pd.DataFrame(
        rows, columns=tuple(_table_contract(G3P_CALIBRATION_SCENARIO_TABLE).columns)
    )
    return _validate_table(G3P_CALIBRATION_SCENARIO_TABLE, frame)


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
        "artifact_schema_version": G3P_CALIBRATION_RESULT_SCHEMA_VERSION,
        "artifact_kind": G3P_CALIBRATION_ARTIFACT_KIND,
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
    "comm_probability_release_allowed",
    "replicate_ids",
    "scenario_result_ids",
    "protocol",
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


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    document = load_schema_document("g3p_calibration_result.schema.json")
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        _fail(
            "G3-P result schema has no properties",
            code="invalid_g3p_calibration_schema",
            field_name="properties",
            remediation="Restore the released G3-P calibration schemas",
        )
    expected_version = _const(
        properties,
        "artifact_schema_version",
        schema="g3p_calibration_result.schema.json",
    )
    expected_kind = _const(
        properties,
        "artifact_kind",
        schema="g3p_calibration_result.schema.json",
    )
    expected_status = _const(
        properties,
        "status",
        schema="g3p_calibration_result.schema.json",
    )
    expected_campaign_version = _const(
        properties,
        "campaign_schema_version",
        schema="g3p_calibration_result.schema.json",
    )
    if set(manifest) != _MANIFEST_FIELDS or (
        manifest.get("artifact_schema_version") != expected_version
        or manifest.get("artifact_kind") != expected_kind
        or manifest.get("status") != expected_status
        or manifest.get("campaign_schema_version") != expected_campaign_version
    ):
        _fail(
            "G3-P calibration manifest violates the current schema",
            code="invalid_g3p_calibration_manifest",
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
                "G3-P calibration manifest has an invalid identifier",
                code="invalid_g3p_calibration_manifest",
                field_name=field_name,
            )
    if manifest["gate_status"] not in {"passed", "failed", "not_estimable"}:
        _fail(
            "G3-P calibration manifest has an invalid gate status",
            code="invalid_g3p_calibration_manifest",
            field_name="gate_status",
        )
    released = manifest["comm_probability_release_allowed"]
    if not isinstance(released, bool) or released != (
        manifest["gate_status"] == "passed"
    ):
        _fail(
            "G3-P release flag differs from its producer-derived gate status",
            code="g3p_calibration_release_mismatch",
            field_name="comm_probability_release_allowed",
        )
    replicate_ids = manifest["replicate_ids"]
    scenario_ids = manifest["scenario_result_ids"]
    if (
        not isinstance(replicate_ids, list)
        or any(not isinstance(item, str) or not item for item in replicate_ids)
        or len(replicate_ids) != len(set(replicate_ids))
        or not isinstance(scenario_ids, list)
        or len(scenario_ids) != 12
        or any(not isinstance(item, str) or not item for item in scenario_ids)
        or len(scenario_ids) != len(set(scenario_ids))
    ):
        _fail(
            "G3-P calibration registries are incomplete or duplicated",
            code="invalid_g3p_calibration_registry",
            field_name="replicate_ids,scenario_result_ids",
        )
    for field_name in ("protocol", "evidence", "calibration_gate"):
        if not isinstance(manifest[field_name], Mapping):
            _fail(
                "G3-P calibration registry is not an object",
                code="invalid_g3p_calibration_registry",
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
                "G3-P generator verification lacks a replay attestation",
                code="g3p_calibration_attestation_mismatch",
                field_name="generator_attestation",
            )
    elif not isinstance(attestation, Mapping) or set(attestation) != (
        _ATTESTATION_FIELDS
    ):
        _fail(
            "G3-P replay attestation has invalid fields",
            code="invalid_g3p_calibration_attestation",
            field_name="generator_attestation",
        )
    else:
        typed_attestation = cast(Mapping[str, Any], attestation)
        attestation_id = typed_attestation.get("attestation_id")
        release_approved = typed_attestation.get("release_approved")
        if (
            not isinstance(attestation_id, str)
            or not attestation_id
            or not isinstance(release_approved, bool)
            or typed_attestation.get("campaign_kind") != "g3_probability"
            or typed_attestation.get("protocol_id") != manifest["protocol_id"]
            or typed_attestation.get("generator_id") != protocol.get("generator_id")
            or evidence.get("generator_attestation_id") != attestation_id
            or gate.get("generator_attestation_id") != attestation_id
            or evidence.get("generator_verified") is not release_approved
            or gate.get("generator_verified") is not release_approved
            or typed_attestation.get("ledger_digest")
            != typed_attestation.get("replayed_ledger_digest")
            or typed_attestation.get("replicate_ids") != manifest["replicate_ids"]
        ):
            _fail(
                "G3-P replay attestation differs from campaign lineage",
                code="g3p_calibration_attestation_mismatch",
                field_name="generator_attestation",
            )
    raw_tables = manifest["tables"]
    if not isinstance(raw_tables, Mapping) or set(raw_tables) != set(_SCHEMA_FILES):
        _fail(
            "G3-P calibration table registry is incomplete",
            code="invalid_g3p_calibration_registry",
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
                "G3-P calibration table record is invalid",
                code="invalid_g3p_calibration_manifest",
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
                "G3-P calibration table metadata is invalid",
                code="invalid_g3p_calibration_manifest",
                field_name=name,
            )
    if manifest["artifact_id"] != stable_id(
        "g3p_calibration_result",
        _manifest_payload(manifest),
        schema_version="1",
    ):
        _fail(
            "G3-P calibration artifact identity is inconsistent",
            code="g3p_calibration_artifact_identity_mismatch",
            field_name="artifact_id",
        )
    return manifest


def write_g3p_calibration_result(
    destination: str | Path,
    *,
    campaign: G3PCalibrationCampaign,
    replay_registry: CalibrationReplayRegistry | None = None,
    n_jobs: int = 1,
) -> G3PCalibrationResult:
    """Atomically persist one replay-verifiable G3-P calibration campaign.

    Metrics and pass/fail values are deliberately absent from the API.  The
    writer accepts only a producer-owned campaign and derives its gate.  A
    campaign carrying an attestation requires the exact registry on write so
    the returned facade is loaded by replay rather than by persisted booleans.
    """

    if not isinstance(campaign, G3PCalibrationCampaign):
        raise TypeError("campaign must be a producer-owned G3PCalibrationCampaign")
    campaign_manifest = campaign.to_manifest()
    if campaign_manifest.get("schema_version") != _CAMPAIGN_SCHEMA_VERSION:
        raise ResultValidationError(
            "G3-P campaign schema is incompatible with this result family",
            code="unsupported_g3p_campaign_schema",
            field="schema_version",
            remediation="Use the current producer to regenerate the campaign",
        )
    if campaign.generator_attestation is not None and replay_registry is None:
        raise ResultValidationError(
            "Attested G3-P campaigns require their replay registry",
            code="g3p_calibration_attestation_unavailable",
            field="replay_registry",
            remediation=(
                "Pass the exact package-owned registry used to replay the campaign"
            ),
        )
    if campaign.generator_attestation is not None:
        if replay_registry is None:
            raise ResultValidationError(
                "Attested G3-P campaign requires its replay registry",
                code="g3p_calibration_attestation_unavailable",
                field="replay_registry",
                remediation="Pass the exact package-owned replay registry",
            )
        try:
            replayed_campaign = summarize_attested_g3p_calibration_campaign(
                campaign.protocol,
                campaign.replicates,
                registry=replay_registry,
                n_jobs=n_jobs,
            )
        except (ContractError, TypeError, ValueError) as error:
            raise ResultValidationError(
                "G3-P campaign could not be replayed before persistence",
                code="g3p_calibration_prewrite_replay_failed",
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
                "Prewrite G3-P replay differs from the supplied campaign",
                code="g3p_calibration_prewrite_replay_mismatch",
                field="campaign_id,generator_attestation",
                remediation="Regenerate the campaign with the exact approved registry",
            )
        campaign = replayed_campaign
        campaign_manifest = campaign.to_manifest()
    gate = build_g3p_calibration_gate(campaign.evidence)
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"G3-P calibration destination {output.name!r} already exists",
            code="g3p_calibration_destination_exists",
            field="destination",
            remediation="Choose a new versioned result directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        tables = {
            G3P_CALIBRATION_REPLICATE_TABLE: _replicate_table(campaign),
            G3P_CALIBRATION_CANDIDATE_TABLE: _candidate_table(campaign),
            G3P_CALIBRATION_SCENARIO_TABLE: _scenario_table(campaign),
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
            "artifact_schema_version": G3P_CALIBRATION_RESULT_SCHEMA_VERSION,
            "artifact_kind": G3P_CALIBRATION_ARTIFACT_KIND,
            "status": _COMPLETE,
            "campaign_schema_version": campaign_manifest["schema_version"],
            "campaign_id": campaign.campaign_id,
            "protocol_id": campaign.protocol.protocol_id,
            "evidence_id": campaign.evidence.evidence_id,
            "gate_id": gate.gate_id,
            "gate_status": gate.status.value,
            "comm_probability_release_allowed": (gate.comm_probability_release_allowed),
            "replicate_ids": [item.replicate_id for item in campaign.replicates],
            "scenario_result_ids": [
                item.scenario_result_id for item in campaign.scenarios
            ],
            "protocol": campaign.protocol.to_dict(),
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
            "g3p_calibration_result",
            manifest,
            schema_version="1",
        )
        _validate_manifest(cast(dict[str, Any], manifest))
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "artifact_schema_version": G3P_CALIBRATION_RESULT_SCHEMA_VERSION,
                "artifact_kind": G3P_CALIBRATION_ARTIFACT_KIND,
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
            "G3-P calibration result write failed and was marked incomplete",
            code="g3p_calibration_write_failed",
            field="destination",
            remediation="Inspect producer diagnostics and write to a new directory",
        ) from error
    return G3PCalibrationResult.load(
        output,
        replay_registry=replay_registry,
        n_jobs=n_jobs,
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
            f"G3-P calibration table {name!r} is missing or corrupted",
            code="corrupted_g3p_calibration_table",
            field=name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if digest != record["sha256"]:
        _fail(
            f"G3-P calibration table {name!r} does not match its manifest",
            code="g3p_calibration_digest_mismatch",
            field_name=name,
        )
    try:
        frame = _read_parquet(path, engine="pyarrow")
    except Exception as error:
        raise ResultValidationError(
            f"G3-P calibration table {name!r} is missing or corrupted",
            code="corrupted_g3p_calibration_table",
            field=name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if len(frame) != record["rows"]:
        _fail(
            f"G3-P calibration table {name!r} row count differs",
            code="g3p_calibration_digest_mismatch",
            field_name=name,
        )
    return _validate_table(name, frame)


def _protocol_from_manifest(manifest: Mapping[str, Any]) -> G3PCalibrationProtocol:
    raw = manifest["protocol"]
    if not isinstance(raw, Mapping):
        _fail(
            "G3-P protocol registry is invalid",
            code="invalid_g3p_calibration_registry",
            field_name="protocol",
        )
    seed_value = raw.get("seed_lineage")
    generator_value = raw.get("generator_manifest")
    if not isinstance(seed_value, Mapping):
        _fail(
            "G3-P protocol seed lineage is invalid",
            code="invalid_g3p_calibration_registry",
            field_name="protocol.seed_lineage",
        )
    if not isinstance(generator_value, Mapping):
        _fail(
            "G3-P generator manifest is invalid",
            code="invalid_g3p_calibration_registry",
            field_name="protocol.generator_manifest",
        )
    try:
        generator_manifest = CalibrationGeneratorManifest.from_dict(
            cast(Mapping[str, object], generator_value)
        )
        protocol = G3PCalibrationProtocol(
            campaign_name=str(raw["campaign_name"]),
            generator_manifest=generator_manifest,
            active_null_spec_id=str(raw["active_null_spec_id"]),
            active_probability_spec_id=str(raw["active_probability_spec_id"]),
            score_spec_id=str(raw["score_spec_id"]),
            candidate_universe_policy_id=str(raw["candidate_universe_policy_id"]),
            estimator_id=str(raw["estimator_id"]),
            stratum_policy_id=str(raw["stratum_policy_id"]),
            score_version=str(raw["score_version"]),
            seed_lineage=SeedLineage.from_dict(cast(Mapping[str, object], seed_value)),
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted G3-P protocol cannot be reconstructed",
            code="invalid_g3p_calibration_protocol",
            field="protocol",
            remediation="Reject the artifact and regenerate it",
        ) from error
    if (
        protocol.to_dict() != dict(raw)
        or raw.get("generator_id") != generator_manifest.generator_id
        or protocol.protocol_id != manifest["protocol_id"]
    ):
        _fail(
            "Persisted G3-P protocol identity differs",
            code="g3p_calibration_protocol_mismatch",
            field_name="protocol_id",
        )
    return protocol


def _parse_seed_lineage(value: object) -> SeedLineage:
    if not isinstance(value, str):
        _fail(
            "G3-P replicate seed lineage is not JSON",
            code="invalid_g3p_calibration_seed_lineage",
            field_name="seed_lineage_json",
        )
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping) or canonical_json(parsed) != value:
            raise ValueError("seed lineage JSON is not canonical")
        return SeedLineage.from_dict(cast(Mapping[str, object], parsed))
    except (ContractError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ResultValidationError(
            "Persisted G3-P replicate seed lineage is invalid",
            code="invalid_g3p_calibration_seed_lineage",
            field="seed_lineage_json",
            remediation="Reject the artifact and regenerate it",
        ) from error


def _replicates_from_tables(
    protocol: G3PCalibrationProtocol,
    replicate_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
) -> tuple[G3PReplicatePredictions, ...]:
    replicate_ids = set(replicate_frame["replicate_id"].astype(str))
    candidate_replicate_ids = set(candidate_frame["replicate_id"].astype(str))
    if candidate_replicate_ids != replicate_ids:
        _fail(
            "G3-P candidate rows do not cover the exact replicate registry",
            code="g3p_calibration_candidate_coverage_mismatch",
            field_name="replicate_id",
        )
    if set(replicate_frame["protocol_id"].astype(str)) not in (
        set(),
        {protocol.protocol_id},
    ) or set(candidate_frame["protocol_id"].astype(str)) not in (
        set(),
        {protocol.protocol_id},
    ):
        _fail(
            "G3-P ledger protocol links differ from the manifest",
            code="g3p_calibration_protocol_link_mismatch",
            field_name="protocol_id",
        )
    reconstructed: list[G3PReplicatePredictions] = []
    for row in replicate_frame.itertuples(index=False):
        source = cast(Any, row)
        replicate_id = str(source.replicate_id)
        candidates = candidate_frame.loc[
            candidate_frame["replicate_id"].astype(str) == replicate_id
        ].sort_values("candidate_index", kind="stable")
        indexes = tuple(int(item) for item in candidates["candidate_index"])
        n_candidates = int(source.n_candidates)
        if indexes != tuple(range(n_candidates)) or len(candidates) != n_candidates:
            _fail(
                "G3-P candidate indexes do not form a complete replicate ledger",
                code="g3p_calibration_candidate_coverage_mismatch",
                field_name="candidate_index",
            )
        expected_scenario_id = protocol.scenario_id(
            str(source.dependence_structure),
            float(source.non_null_prevalence),
        )
        if str(source.scenario_id) != expected_scenario_id or set(
            candidates["scenario_id"].astype(str)
        ) != {expected_scenario_id}:
            _fail(
                "G3-P replicate scenario links differ from the protocol",
                code="g3p_calibration_scenario_link_mismatch",
                field_name="scenario_id",
            )
        candidate_ids = tuple(candidates["candidate_edge_id"].astype(str))
        strata = tuple(candidates["stratum_id"].astype(str))
        truth = np.asarray(candidates["truth_active"], dtype="|b1")
        probabilities = np.asarray(
            [
                np.nan if _is_missing(value) else float(value)
                for value in candidates["candidate_probability"].array
            ],
            dtype="<f8",
        )
        candidate_statuses = tuple(candidates["candidate_status"].astype(str))
        candidate_reason_codes = tuple(
            None if _is_missing(value) else str(value)
            for value in candidates["candidate_reason_code"].array
        )
        source_collection = (
            None
            if _is_missing(source.source_probability_collection_id)
            else str(source.source_probability_collection_id)
        )
        reason = None if _is_missing(source.reason_code) else str(source.reason_code)
        try:
            replicate = G3PReplicatePredictions(
                protocol_id=str(source.protocol_id),
                dependence_structure=str(source.dependence_structure),
                non_null_prevalence=float(source.non_null_prevalence),
                replicate_index=int(source.replicate_index),
                seed_lineage=_parse_seed_lineage(source.seed_lineage_json),
                source_active_null_id=str(source.source_active_null_id),
                source_candidate_universe_id=str(source.source_candidate_universe_id),
                source_probability_collection_id=source_collection,
                candidate_edge_ids=candidate_ids,
                stratum_ids=strata,
                truth_active=truth,
                candidate_probabilities=probabilities,
                candidate_statuses=candidate_statuses,
                candidate_reason_codes=candidate_reason_codes,
                status=str(source.status),
                reason_code=reason,
            )
        except (ContractError, TypeError, ValueError) as error:
            raise ResultValidationError(
                "Persisted G3-P replicate cannot be reconstructed",
                code="invalid_g3p_calibration_replicate",
                field="replicate_id",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if replicate.replicate_id != replicate_id:
            _fail(
                "Persisted G3-P replicate identity cannot be reconstructed",
                code="g3p_calibration_replicate_identity_mismatch",
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


_STRATUM_RESULT_FIELDS = {
    "stratum_result_id",
    "scenario_id",
    "protocol_id",
    "stratum_id",
    "status",
    "reason_code",
    "n_replicates",
    "minimum_eligible_candidates",
    "observed_prevalence",
    "brier_score",
    "prevalence_only_brier_score",
    "brier_relative_improvement",
    "ece",
    "ece_upper_bound",
    "calibration_in_the_large",
    "calibration_slope",
    "bootstrap_seed",
    "metric_weighting",
}


def _parse_canonical_json_array(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, str):
        _fail(
            "G3-P scenario registry is not canonical JSON",
            code="invalid_g3p_calibration_scenario_registry",
            field_name=field_name,
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ResultValidationError(
            "G3-P scenario registry contains invalid JSON",
            code="invalid_g3p_calibration_scenario_registry",
            field=field_name,
            remediation="Reject the artifact and regenerate it from intact ledgers",
        ) from error
    if not isinstance(parsed, list) or canonical_json(parsed) != value:
        _fail(
            "G3-P scenario registry is not a canonical JSON array",
            code="invalid_g3p_calibration_scenario_registry",
            field_name=field_name,
        )
    return parsed


def _validate_scenario_json_links(frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        source = cast(Any, row)
        replicate_ids = _parse_canonical_json_array(
            source.replicate_ids_json,
            field_name="replicate_ids_json",
        )
        stratum_result_ids = _parse_canonical_json_array(
            source.stratum_result_ids_json,
            field_name="stratum_result_ids_json",
        )
        strata = _parse_canonical_json_array(
            source.strata_json,
            field_name="strata_json",
        )
        if (
            any(not isinstance(item, str) or not item for item in replicate_ids)
            or len(replicate_ids) != len(set(replicate_ids))
            or any(not isinstance(item, str) or not item for item in stratum_result_ids)
            or len(stratum_result_ids) != len(set(stratum_result_ids))
            or any(not isinstance(item, Mapping) for item in strata)
        ):
            _fail(
                "G3-P scenario JSON registries are malformed or duplicated",
                code="invalid_g3p_calibration_scenario_registry",
                field_name="replicate_ids_json,stratum_result_ids_json,strata_json",
            )
        typed_strata = cast(list[Mapping[str, Any]], strata)
        if (
            any(set(item) != _STRATUM_RESULT_FIELDS for item in typed_strata)
            or any(
                not isinstance(item["stratum_result_id"], str)
                or not item["stratum_result_id"]
                or not isinstance(item["scenario_id"], str)
                or not item["scenario_id"]
                or not isinstance(item["protocol_id"], str)
                or not item["protocol_id"]
                or not isinstance(item["stratum_id"], str)
                or not item["stratum_id"]
                for item in typed_strata
            )
            or [item["stratum_result_id"] for item in typed_strata]
            != stratum_result_ids
            or len({item["stratum_id"] for item in typed_strata}) != len(typed_strata)
            or any(
                item["scenario_id"] != str(source.scenario_id)
                or item["protocol_id"] != str(source.protocol_id)
                for item in typed_strata
            )
            or [item["stratum_id"] for item in typed_strata]
            != sorted(item["stratum_id"] for item in typed_strata)
        ):
            _fail(
                "G3-P per-stratum JSON links differ from their scenario row",
                code="g3p_calibration_stratum_link_mismatch",
                field_name="stratum_result_ids_json,strata_json",
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
            f"Persisted G3-P table {name!r} differs from producer reconstruction",
            code="g3p_calibration_reconstruction_mismatch",
            field_name=name,
        )


def _validate_links(
    manifest: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    *,
    replay_registry: CalibrationReplayRegistry | None,
    n_jobs: int,
) -> tuple[G3PCalibrationCampaign, G3PCalibrationGate]:
    _validate_scenario_json_links(tables[G3P_CALIBRATION_SCENARIO_TABLE])
    protocol = _protocol_from_manifest(manifest)
    replicates = _replicates_from_tables(
        protocol,
        tables[G3P_CALIBRATION_REPLICATE_TABLE],
        tables[G3P_CALIBRATION_CANDIDATE_TABLE],
    )
    if manifest["generator_attestation"] is not None and replay_registry is None:
        raise ResultValidationError(
            "Attested G3-P artifact requires its replay registry",
            code="g3p_calibration_attestation_unavailable",
            field="replay_registry",
            remediation="Load with the exact package-owned replay registry",
        )
    try:
        if manifest["generator_attestation"] is None:
            campaign = summarize_g3p_calibration_campaign(protocol, replicates)
        else:
            if replay_registry is None:
                raise ResultValidationError(
                    "Attested G3-P artifact requires its replay registry",
                    code="g3p_calibration_attestation_unavailable",
                    field="replay_registry",
                    remediation="Load with the exact package-owned replay registry",
                )
            campaign = summarize_attested_g3p_calibration_campaign(
                protocol,
                replicates,
                registry=replay_registry,
                n_jobs=n_jobs,
            )
        gate = build_g3p_calibration_gate(campaign.evidence)
    except (ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted G3-P ledgers do not form a valid campaign",
            code="invalid_g3p_calibration_campaign",
            field="replicates,candidates",
            remediation="Reject the artifact and regenerate it",
        ) from error
    expected_tables = {
        G3P_CALIBRATION_REPLICATE_TABLE: _replicate_table(campaign),
        G3P_CALIBRATION_CANDIDATE_TABLE: _candidate_table(campaign),
        G3P_CALIBRATION_SCENARIO_TABLE: _scenario_table(campaign),
    }
    for name, expected in expected_tables.items():
        _require_exact_table(name, tables[name], expected)
    expected_replicate_ids = [item.replicate_id for item in campaign.replicates]
    expected_scenario_ids = [item.scenario_result_id for item in campaign.scenarios]
    if (
        manifest["campaign_id"] != campaign.campaign_id
        or manifest["protocol_id"] != campaign.protocol.protocol_id
        or manifest["evidence_id"] != campaign.evidence.evidence_id
        or manifest["gate_id"] != gate.gate_id
        or manifest["gate_status"] != gate.status.value
        or manifest["comm_probability_release_allowed"]
        != gate.comm_probability_release_allowed
        or manifest["replicate_ids"] != expected_replicate_ids
        or manifest["scenario_result_ids"] != expected_scenario_ids
        or manifest["protocol"] != campaign.protocol.to_dict()
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
            "G3-P campaign, evidence, or gate linkage differs from reconstruction",
            code="g3p_calibration_linkage_mismatch",
            field_name="campaign_id,evidence_id,gate_id",
        )
    return campaign, gate


@dataclass(frozen=True, slots=True, init=False)
class G3PCalibrationResult:
    """Immutable facade over one independent G3-P campaign directory."""

    path: Path
    _manifest: dict[str, Any] = field(repr=False)
    _calibration_gate: G3PCalibrationGate = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "G3PCalibrationResult is producer-owned; use "
            "write_g3p_calibration_result() or load()"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        replay_registry: CalibrationReplayRegistry | None = None,
        n_jobs: int = 1,
    ) -> G3PCalibrationResult:
        """Load a completed artifact, replaying attested campaigns exactly."""

        root = Path(path)
        try:
            marker = _read_json(root / _STATUS_FILENAME)
        except Exception as error:
            raise ResultValidationError(
                "G3-P calibration status marker is missing or corrupted",
                code="invalid_g3p_calibration_status",
                field="path",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if marker.get("status") != _COMPLETE:
            raise IncompleteResultError(
                "G3-P calibration result is incomplete",
                code="incomplete_g3p_calibration_result",
                field="path",
                remediation="Inspect producer diagnostics and rerun to a new directory",
            )
        if set(marker) != {
            "artifact_schema_version",
            "artifact_kind",
            "status",
            "artifact_id",
        } or (
            marker.get("artifact_schema_version")
            != G3P_CALIBRATION_RESULT_SCHEMA_VERSION
            or marker.get("artifact_kind") != G3P_CALIBRATION_ARTIFACT_KIND
        ):
            _fail(
                "G3-P calibration status marker violates the current schema",
                code="invalid_g3p_calibration_status",
                field_name="path",
            )
        try:
            manifest = _validate_manifest(_read_json(root / _MANIFEST_FILENAME))
        except ResultValidationError:
            raise
        except Exception as error:
            raise ResultValidationError(
                "G3-P calibration manifest is missing or corrupted",
                code="invalid_g3p_calibration_manifest",
                field="path",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if marker["artifact_id"] != manifest["artifact_id"]:
            _fail(
                "G3-P calibration status and manifest identities differ",
                code="g3p_calibration_status_manifest_mismatch",
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
            n_jobs=n_jobs,
        )
        self = object.__new__(cls)
        object.__setattr__(self, "path", root.resolve())
        object.__setattr__(self, "_manifest", copy.deepcopy(manifest))
        object.__setattr__(self, "_calibration_gate", gate)
        return self

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive copy of the complete campaign registry."""

        return copy.deepcopy(self._manifest)

    @property
    def calibration_gate(self) -> G3PCalibrationGate:
        """Return the gate re-derived from persisted raw ledgers."""

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
        """Read the complete replicate registry."""

        return self._read(
            G3P_CALIBRATION_REPLICATE_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_candidates(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read complete candidate truth and probability ledgers."""

        return self._read(
            G3P_CALIBRATION_CANDIDATE_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_scenarios(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read producer-derived scenario metrics without recalculation."""

        return self._read(
            G3P_CALIBRATION_SCENARIO_TABLE,
            filters=filters,
            columns=columns,
        )


__all__ = [
    "G3P_CALIBRATION_ARTIFACT_KIND",
    "G3P_CALIBRATION_CANDIDATE_TABLE",
    "G3P_CALIBRATION_REPLICATE_TABLE",
    "G3P_CALIBRATION_RESULT_SCHEMA_VERSION",
    "G3P_CALIBRATION_SCENARIO_TABLE",
    "G3PCalibrationResult",
    "write_g3p_calibration_result",
]
