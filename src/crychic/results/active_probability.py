"""Independent ADR-013 active-probability result artifact.

This result family deliberately does not extend the exploratory v0.1 result
manifest or the descriptive cross-fit v1-v5 bundles. Candidate empirical
p-values and local-FDR values remain in diagnostic tables. A public
``comm_probability`` table exists only when the producer-owned G3-P gate bound
to the exact runtime contracts passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, NoReturn, cast

import pandas as pd

from crychic.core import CommunicationMode, ContractError, canonical_json, stable_id
from crychic.inference import (
    ActiveEdgeScoreStatus,
    ActiveProbabilityCollection,
    ActiveProbabilityRecord,
    ActiveProbabilitySpec,
    G3PCalibrationGate,
    G3PGateStatus,
    NullActiveEdgeScoreRecord,
    NullScoreDistribution,
    PointActiveEdgeScoreRecord,
    estimate_active_probabilities,
)
from crychic.inference.calibration_attestation import CalibrationReplayRegistry
from crychic.scoring import (
    ActiveEdgeCandidate,
    FrozenActiveEdgeUniverse,
    freeze_active_edge_universe,
)

from ._schema import load_schema_document
from .errors import (
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
)
from .g3p_calibration import G3PCalibrationResult

ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION = "3.0.0"
_ACTIVE_PROBABILITY_TABLE_SCHEMA_VERSION = "1.0.0"
ACTIVE_PROBABILITY_ARTIFACT_KIND = "adr013_active_probability_result"

ACTIVE_PROBABILITY_TABLE = "active_probabilities"
ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE = (
    "active_probability_candidate_diagnostics"
)
ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE = "active_probability_stratum_diagnostics"
ACTIVE_PROBABILITY_NULL_SOURCE_TABLE = "active_probability_null_sources"

_STATUS_FILENAME = "_status.json"
_MANIFEST_FILENAME = "active_probability_manifest.json"
_COMPLETE = "complete"
_INCOMPLETE = "incomplete"
_RELEASED = "released"
_DIAGNOSTIC_ONLY = "diagnostic_only"
_COLLECTION_PRODUCER = "crychic.inference.active_probability.collection.v1"
_FIT_PRODUCER = "crychic.inference.active_probability.bum_fit.v1"
_GATE_PRODUCER = "crychic.inference.active_probability.g3p_gate.v3"
_EMPIRICAL_P_SEMANTICS = "matched_edge_right_tail_plus_one_v1"
_PROBABILITY_SEMANTICS = "one_minus_beta_uniform_mixture_local_fdr_v1"
_BUM_ESTIMATOR = "beta_uniform_mixture_pi0_0.5_1_a_0.05_0.95_v1"
_ARTIFACT_ID_SCHEMA_VERSION = "2"

_SCHEMA_FILES = {
    ACTIVE_PROBABILITY_TABLE: "active_probabilities.schema.json",
    ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE: (
        "active_probability_candidate_diagnostics.schema.json"
    ),
    ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE: (
        "active_probability_stratum_diagnostics.schema.json"
    ),
    ACTIVE_PROBABILITY_NULL_SOURCE_TABLE: (
        "active_probability_null_sources.schema.json"
    ),
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
    columns: Mapping[str, _ColumnContract]


def _fail(
    message: str,
    *,
    code: str,
    field_name: str,
    remediation: str = "Reject the artifact and regenerate it from intact parents",
) -> NoReturn:
    raise ResultValidationError(
        message,
        code=code,
        field=field_name,
        remediation=remediation,
    )


def _const(properties: Mapping[str, Any], name: str, *, schema: str) -> Any:
    raw = properties.get(name)
    if not isinstance(raw, Mapping) or "const" not in raw:
        _fail(
            f"Active-probability schema {schema!r} lacks {name!r}",
            code="invalid_active_probability_schema",
            field_name=name,
            remediation="Restore the released active-probability schemas",
        )
    return raw["const"]


@cache
def _table_contract(name: str) -> _TableContract:
    if name not in _SCHEMA_FILES:
        raise KeyError(name)
    schema_filename = _SCHEMA_FILES[name]
    document = load_schema_document(schema_filename)
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        _fail(
            f"Active-probability schema {schema_filename!r} has no properties",
            code="invalid_active_probability_schema",
            field_name="properties",
            remediation="Restore the released active-probability schemas",
        )
    version = _const(properties, "artifact_schema_version", schema=schema_filename)
    table_name = _const(properties, "table", schema=schema_filename)
    filename = _const(properties, "filename", schema=schema_filename)
    primary_key = _const(properties, "primary_key", schema=schema_filename)
    column_block = properties.get("columns")
    raw_columns = (
        column_block.get("properties") if isinstance(column_block, Mapping) else None
    )
    if (
        version != _ACTIVE_PROBABILITY_TABLE_SCHEMA_VERSION
        or table_name != name
        or not isinstance(filename, str)
        or not isinstance(primary_key, list)
        or not isinstance(raw_columns, Mapping)
    ):
        _fail(
            f"Active-probability schema {schema_filename!r} is incompatible",
            code="invalid_active_probability_schema",
            field_name="artifact_schema_version",
            remediation="Use schemas from this artifact implementation",
        )
    columns: dict[str, _ColumnContract] = {}
    for column_name, raw in raw_columns.items():
        if not isinstance(column_name, str) or not isinstance(raw, Mapping):
            _fail(
                "Active-probability table has an invalid column contract",
                code="invalid_active_probability_schema",
                field_name="columns",
            )
        dtype = raw.get("dtype")
        nullable = raw.get("nullable")
        enum = raw.get("enum", [])
        if (
            dtype not in {"string", "float", "integer", "boolean"}
            or not isinstance(nullable, bool)
            or not isinstance(enum, list)
        ):
            _fail(
                f"Active-probability column {column_name!r} is invalid",
                code="invalid_active_probability_schema",
                field_name=column_name,
            )
        columns[column_name] = _ColumnContract(
            dtype=dtype,
            nullable=nullable,
            enum=tuple(enum),
            minimum=cast(float | None, raw.get("minimum")),
            maximum=cast(float | None, raw.get("maximum")),
        )
    if not set(primary_key).issubset(columns):
        _fail(
            "Active-probability primary key is absent from its columns",
            code="invalid_active_probability_schema",
            field_name="primary_key",
        )
    return _TableContract(
        name=name,
        filename=filename,
        schema_filename=schema_filename,
        primary_key=tuple(cast(list[str], primary_key)),
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
                code="invalid_active_probability_null",
                field_name=column_name,
            )
        return
    if contract.dtype == "string":
        if not isinstance(value, str) or not value:
            _fail(
                f"{table_name}.{column_name} must be a non-empty string",
                code="invalid_active_probability_dtype",
                field_name=column_name,
            )
        normalized: object = value
    elif contract.dtype == "boolean":
        if type(value) is not bool:
            _fail(
                f"{table_name}.{column_name} must be boolean",
                code="invalid_active_probability_dtype",
                field_name=column_name,
            )
        normalized = value
    elif contract.dtype == "integer":
        if isinstance(value, bool):
            _fail(
                f"{table_name}.{column_name} must be integer",
                code="invalid_active_probability_dtype",
                field_name=column_name,
            )
        try:
            numeric = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            numeric = math.nan
        if not math.isfinite(numeric) or not numeric.is_integer():
            _fail(
                f"{table_name}.{column_name} must be integer",
                code="invalid_active_probability_dtype",
                field_name=column_name,
            )
        normalized = numeric
    else:
        if isinstance(value, bool):
            _fail(
                f"{table_name}.{column_name} must be numeric",
                code="invalid_active_probability_dtype",
                field_name=column_name,
            )
        try:
            numeric = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            numeric = math.nan
        if not math.isfinite(numeric):
            _fail(
                f"{table_name}.{column_name} must be finite",
                code="invalid_active_probability_dtype",
                field_name=column_name,
            )
        normalized = numeric
    if contract.enum and normalized not in contract.enum:
        _fail(
            f"{table_name}.{column_name} has an unsupported value",
            code="invalid_active_probability_enum",
            field_name=column_name,
        )
    if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
        if contract.minimum is not None and normalized < contract.minimum:
            _fail(
                f"{table_name}.{column_name} is below its minimum",
                code="invalid_active_probability_range",
                field_name=column_name,
            )
        if contract.maximum is not None and normalized > contract.maximum:
            _fail(
                f"{table_name}.{column_name} exceeds its maximum",
                code="invalid_active_probability_range",
                field_name=column_name,
            )


def _validate_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("active-probability tables must be pandas DataFrames")
    contract = _table_contract(name)
    if tuple(frame.columns) != tuple(contract.columns):
        _fail(
            f"Active-probability table {name!r} columns do not match its schema",
            code="invalid_active_probability_columns",
            field_name=name,
        )
    result = frame.copy(deep=True)
    for column_name, column_contract in contract.columns.items():
        for value in result[column_name].tolist():
            _validate_scalar(
                value,
                column_contract,
                table_name=name,
                column_name=column_name,
            )
    if result.duplicated(list(contract.primary_key)).any():
        _fail(
            f"Active-probability table {name!r} has duplicate keys",
            code="duplicate_active_probability_key",
            field_name=contract.primary_key[0],
        )
    if name == ACTIVE_PROBABILITY_TABLE:
        released = result["status"].eq(_RELEASED)
        if (
            result.loc[released, "comm_probability"].isna().any()
            or result.loc[released, "reason_code"].notna().any()
            or result.loc[~released, "comm_probability"].notna().any()
            or result.loc[~released, "reason_code"].isna().any()
        ):
            _fail(
                "Public active-probability rows have invalid value/reason semantics",
                code="invalid_active_probability_release_state",
                field_name="comm_probability,status,reason_code",
            )
    elif name == ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE:
        both = (
            result["candidate_local_fdr"]
            .isna()
            .eq(result["candidate_comm_probability"].isna())
        )
        available = result["candidate_local_fdr"].notna()
        sums = result.loc[available, "candidate_local_fdr"].astype(float) + result.loc[
            available, "candidate_comm_probability"
        ].astype(float)
        if not bool(both.all()) or not bool(
            sums.map(lambda value: math.isclose(value, 1.0, abs_tol=1e-12)).all()
        ):
            _fail(
                "Candidate local-FDR and probability diagnostics are inconsistent",
                code="invalid_active_probability_diagnostic_state",
                field_name="candidate_local_fdr,candidate_comm_probability",
            )
    return result


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
        "artifact_schema_version": ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION,
        "artifact_kind": ACTIVE_PROBABILITY_ARTIFACT_KIND,
        "status": _INCOMPLETE,
    }
    if error_type is not None:
        marker["error_type"] = error_type
    try:
        _write_json(path / _STATUS_FILENAME, marker)
    except OSError:
        pass


def _require_parents(
    collection: ActiveProbabilityCollection,
    distribution: NullScoreDistribution,
    universe: FrozenActiveEdgeUniverse,
    spec: ActiveProbabilitySpec,
    gate: G3PCalibrationGate | None,
) -> bool:
    if not isinstance(collection, ActiveProbabilityCollection):
        raise TypeError("collection must be ActiveProbabilityCollection")
    if not isinstance(distribution, NullScoreDistribution):
        raise TypeError("distribution must be NullScoreDistribution")
    if not isinstance(universe, FrozenActiveEdgeUniverse):
        raise TypeError("universe must be FrozenActiveEdgeUniverse")
    if not isinstance(spec, ActiveProbabilitySpec):
        raise TypeError("spec must be ActiveProbabilitySpec")
    if gate is not None and not isinstance(gate, G3PCalibrationGate):
        raise TypeError("calibration_gate must be G3PCalibrationGate or None")
    collection._require_intact()
    distribution._require_intact()
    universe._require_intact()
    spec._require_intact()
    if gate is not None:
        gate._require_intact()
    collection_edge_ids = tuple(item.candidate_edge_id for item in collection.records)
    point_ids = tuple(item.candidate_edge_id for item in distribution.point_records)
    expected_gate_id = None if gate is None else gate.gate_id
    expected_gate_status = None if gate is None else gate.status
    if (
        collection.distribution_id != distribution.distribution_id
        or collection.active_null_id != distribution.active_null_id
        or collection.candidate_universe_id != universe.universe_id
        or distribution.candidate_universe_id != universe.universe_id
        or collection.score_spec_id != distribution.score_spec_id
        or collection.active_probability_spec_id != spec.spec_id
        or collection.calibration_gate_id != expected_gate_id
        or collection.calibration_gate_status != expected_gate_status
        or collection_edge_ids != universe.candidate_edge_ids
        or point_ids != universe.candidate_edge_ids
        or distribution.score_version != universe.score_version
    ):
        _fail(
            "Active-probability parents do not form one exact runtime contract",
            code="active_probability_parent_binding_mismatch",
            field_name=("collection,distribution,universe,spec,calibration_gate"),
        )
    if gate is not None and (
        gate.active_null_spec_id != distribution.active_null_spec_id
        or gate.active_probability_spec_id != spec.spec_id
        or gate.score_spec_id != distribution.score_spec_id
        or gate.candidate_universe_policy_id
        != distribution.candidate_universe_policy_id
        or distribution.candidate_universe_policy_id
        != universe.candidate_universe_policy_id
        or gate.estimator_id != spec.estimator_id
        or gate.stratum_policy_id != spec.stratum_policy_id
        or gate.score_version != distribution.score_version
    ):
        _fail(
            "G3-P gate is not bound to the persisted runtime contracts",
            code="active_probability_gate_binding_mismatch",
            field_name="calibration_gate_id",
        )
    released = bool(
        gate is not None
        and gate.status is G3PGateStatus.PASSED
        and collection.comm_probability_release_allowed
    )
    if released != collection.comm_probability_release_allowed:
        _fail(
            "Active-probability release state differs from the bound G3-P gate",
            code="active_probability_release_state_mismatch",
            field_name="comm_probability_release_allowed",
        )
    if not released and any(
        item.comm_probability is not None for item in collection.records
    ):
        _fail(
            "Diagnostic-only collection contains a public probability",
            code="active_probability_release_not_authorized",
            field_name="comm_probability",
        )
    return released


def _candidate_diagnostics(
    collection: ActiveProbabilityCollection,
    distribution: NullScoreDistribution,
) -> pd.DataFrame:
    points = {item.candidate_edge_id: item for item in distribution.point_records}
    rows = []
    for record in collection.records:
        point = points[record.candidate_edge_id]
        if record.point_record_id != point.record_id:
            _fail(
                "Active-probability record points to the wrong point score",
                code="active_probability_point_binding_mismatch",
                field_name="point_record_id",
            )
        rows.append(
            {
                "candidate_edge_id": record.candidate_edge_id,
                "active_probability_record_id": record.record_id,
                "point_record_id": record.point_record_id,
                "source_score_collection_id": point.source_score_collection_id,
                "stratum_id": record.stratum_id,
                "score_version": record.score_version,
                "n_subjects": point.n_subjects,
                "point_score": record.point_score,
                "point_status": record.point_status.value,
                "point_reason_code": record.point_reason_code,
                "active_null_empirical_p_value": (record.active_null_empirical_p_value),
                "candidate_local_fdr": record.candidate_local_fdr,
                "candidate_comm_probability": (record.candidate_comm_probability),
                "probability_reason_code": record.probability_reason_code,
                "local_fdr_diagnostic_id": record.local_fdr_diagnostic_id,
                "calibration_gate_id": record.calibration_gate_id,
            }
        )
    contract = _table_contract(ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE)
    return _validate_table(
        ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE,
        pd.DataFrame(rows, columns=tuple(contract.columns)),
    )


def _stratum_diagnostics(collection: ActiveProbabilityCollection) -> pd.DataFrame:
    rows = [
        {
            "local_fdr_diagnostic_id": item.diagnostic_id,
            "stratum_id": item.stratum_id,
            "status": item.status.value,
            "reason_code": item.reason_code,
            "n_candidate_edges": item.n_candidate_edges,
            "n_null_plans": item.n_null_plans,
            "complete_score_matrix": item.complete_score_matrix,
            "pi0": item.pi0,
            "a": item.a,
            "log_likelihood": item.log_likelihood,
            "projected_gradient_norm": item.projected_gradient_norm,
            "curvature_minimum_eigenvalue": (item.curvature_minimum_eigenvalue),
            "n_starts": item.n_starts,
            "n_successful_starts": item.n_successful_starts,
            "log_likelihood_spread": item.log_likelihood_spread,
        }
        for item in collection.diagnostics
    ]
    contract = _table_contract(ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE)
    return _validate_table(
        ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE,
        pd.DataFrame(rows, columns=tuple(contract.columns)),
    )


def _null_sources(distribution: NullScoreDistribution) -> pd.DataFrame:
    rows = [
        {
            "candidate_edge_id": item.candidate_edge_id,
            "plan_id": item.plan_id,
            "stratum_id": item.stratum_id,
            "score_version": item.score_version,
            "null_rerun_record_id": item.null_rerun_record_id,
            "source_score_collection_id": item.source_score_collection_id,
            "n_subjects": item.n_subjects,
            "score": item.score,
            "status": item.status.value,
            "reason_code": item.reason_code,
            "null_record_id": item.record_id,
        }
        for item in distribution.null_records
    ]
    contract = _table_contract(ACTIVE_PROBABILITY_NULL_SOURCE_TABLE)
    return _validate_table(
        ACTIVE_PROBABILITY_NULL_SOURCE_TABLE,
        pd.DataFrame(rows, columns=tuple(contract.columns)),
    )


def _released_probabilities(
    collection: ActiveProbabilityCollection,
    universe: FrozenActiveEdgeUniverse,
) -> pd.DataFrame:
    records = {item.candidate_edge_id: item for item in collection.records}
    rows = []
    for candidate in universe.candidates:
        record = records[candidate.candidate_edge_id]
        is_released = record.comm_probability is not None
        rows.append(
            {
                "candidate_edge_id": candidate.candidate_edge_id,
                "contrast_id": candidate.contrast_id,
                "context_id": candidate.context_id,
                "sender": candidate.sender,
                "receiver": candidate.receiver,
                "interaction_id": candidate.interaction_id,
                "driver_id": candidate.driver_id,
                "mode": CommunicationMode(candidate.mode).value,
                "stratum_id": record.stratum_id,
                "score_version": record.score_version,
                "comm_probability": record.comm_probability,
                "status": _RELEASED if is_released else "not_estimable",
                "reason_code": None if is_released else record.probability_reason_code,
                "active_probability_record_id": record.record_id,
            }
        )
    contract = _table_contract(ACTIVE_PROBABILITY_TABLE)
    return _validate_table(
        ACTIVE_PROBABILITY_TABLE,
        pd.DataFrame(rows, columns=tuple(contract.columns)),
    )


def _table_record(
    path: Path, table: pd.DataFrame, contract: _TableContract
) -> dict[str, object]:
    return {
        "filename": contract.filename,
        "rows": len(table),
        "sha256": _sha256_file(path),
        "schema": contract.schema_filename,
    }


def _collection_id_from_manifest(manifest: Mapping[str, Any]) -> str:
    payload = {
        "distribution_id": manifest["distribution_id"],
        "active_null_id": manifest["active_null_id"],
        "candidate_universe_id": manifest["candidate_universe_id"],
        "score_spec_id": manifest["score_spec_id"],
        "active_probability_spec_id": manifest["active_probability_spec_id"],
        "calibration_gate_id": manifest["calibration_gate_id"],
        "calibration_gate_status": manifest["calibration_gate_status"],
        "record_ids": manifest["active_probability_record_ids"],
        "diagnostic_ids": manifest["local_fdr_diagnostic_ids"],
        "status_counts": manifest["status_counts"],
        "complete_candidate_coverage": True,
        "comm_probability_release_allowed": manifest[
            "comm_probability_release_allowed"
        ],
        "producer_marker": _COLLECTION_PRODUCER,
    }
    identifier: str = stable_id(
        "active_probability_collection",
        payload,
        schema_version="1.0.0",
    )
    return identifier


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, object]:
    return {str(key): value for key, value in manifest.items() if key != "artifact_id"}


def _require_replayed_calibration_result(
    calibration_result: G3PCalibrationResult | None,
    calibration_gate: G3PCalibrationGate,
    *,
    replay_registry: CalibrationReplayRegistry | None,
    n_jobs: int,
) -> G3PCalibrationResult:
    if calibration_result is None:
        raise ResultValidationError(
            "Released probabilities require the authoritative G3-P result artifact",
            code="active_probability_calibration_result_unavailable",
            field="calibration_result",
            remediation=(
                "Pass the G3PCalibrationResult whose replayed gate authorized release"
            ),
        )
    if not isinstance(calibration_result, G3PCalibrationResult):
        raise TypeError("calibration_result must be G3PCalibrationResult or None")
    if replay_registry is None:
        raise ResultValidationError(
            "Released probabilities require the G3-P replay registry",
            code="active_probability_calibration_replay_unavailable",
            field="replay_registry",
            remediation="Pass the exact registry used to replay the G3-P raw ledger",
        )
    replayed = G3PCalibrationResult.load(
        calibration_result.path,
        replay_registry=replay_registry,
        n_jobs=n_jobs,
    )
    parent = replayed.manifest
    replayed_gate = replayed.calibration_gate
    if (
        parent["comm_probability_release_allowed"] is not True
        or parent["gate_status"] != "passed"
        or parent["generator_attestation"] is None
        or parent["calibration_gate"] != replayed_gate.to_dict()
        or replayed_gate.gate_id != calibration_gate.gate_id
        or replayed_gate.to_dict() != calibration_gate.to_dict()
    ):
        _fail(
            "G3-P result artifact does not authorize the supplied release gate",
            code="active_probability_calibration_result_binding_mismatch",
            field_name="calibration_result,calibration_gate",
        )
    return replayed


def write_active_probability_result(
    destination: str | Path,
    *,
    collection: ActiveProbabilityCollection,
    distribution: NullScoreDistribution,
    universe: FrozenActiveEdgeUniverse,
    spec: ActiveProbabilitySpec,
    calibration_gate: G3PCalibrationGate | None,
    calibration_result: G3PCalibrationResult | None = None,
    replay_registry: CalibrationReplayRegistry | None = None,
    n_jobs: int = 1,
) -> ActiveProbabilityResult:
    """Atomically persist one diagnostic-only or G3-P-released artifact.

    Release mode is derived from the producer-owned collection and gate. There
    is intentionally no caller-controlled release switch or probability input.
    """

    released = _require_parents(
        collection,
        distribution,
        universe,
        spec,
        calibration_gate,
    )
    replayed_calibration: G3PCalibrationResult | None = None
    if released:
        if calibration_gate is None:
            raise ResultValidationError(
                "Released active probability requires a G3-P calibration gate",
                code="active_probability_calibration_gate_unavailable",
                field="calibration_gate",
                remediation="Rebuild the release from an intact producer-owned gate",
            )
        replayed_calibration = _require_replayed_calibration_result(
            calibration_result,
            calibration_gate,
            replay_registry=replay_registry,
            n_jobs=n_jobs,
        )
    elif calibration_result is not None:
        raise ResultValidationError(
            "Diagnostic active-probability results cannot claim a G3-P result parent",
            code="active_probability_calibration_result_unexpected",
            field="calibration_result",
            remediation="Omit calibration_result for diagnostic-only persistence",
        )
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"Active-probability destination {output.name!r} already exists",
            code="active_probability_destination_exists",
            field="destination",
            remediation="Choose a new versioned result directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        tables = {
            ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE: (
                _candidate_diagnostics(collection, distribution)
            ),
            ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE: (
                _stratum_diagnostics(collection)
            ),
            ACTIVE_PROBABILITY_NULL_SOURCE_TABLE: _null_sources(distribution),
        }
        if released:
            tables[ACTIVE_PROBABILITY_TABLE] = _released_probabilities(
                collection, universe
            )
        records: dict[str, dict[str, object]] = {}
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
            records[name] = _table_record(path, table, contract)
        manifest: dict[str, object] = {
            "artifact_schema_version": ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION,
            "artifact_kind": ACTIVE_PROBABILITY_ARTIFACT_KIND,
            "status": _COMPLETE,
            "release_mode": _RELEASED if released else _DIAGNOSTIC_ONLY,
            "comm_probability_release_allowed": released,
            "collection_id": collection.collection_id,
            "distribution_id": distribution.distribution_id,
            "active_null_id": distribution.active_null_id,
            "active_null_spec_id": distribution.active_null_spec_id,
            "candidate_universe_id": universe.universe_id,
            "candidate_universe_policy_id": (distribution.candidate_universe_policy_id),
            "score_spec_id": distribution.score_spec_id,
            "active_probability_spec_id": spec.spec_id,
            "calibration_gate_id": (
                None if calibration_gate is None else calibration_gate.gate_id
            ),
            "calibration_gate_status": (
                None if calibration_gate is None else calibration_gate.status.value
            ),
            "calibration_result_artifact_id": (
                None
                if replayed_calibration is None
                else replayed_calibration.manifest["artifact_id"]
            ),
            "calibration_campaign_artifact_id": (
                None
                if calibration_gate is None
                else calibration_gate.source_artifact_id
            ),
            "calibration_evidence_id": (
                None if calibration_gate is None else calibration_gate.evidence_id
            ),
            "calibration_protocol_id": (
                None if calibration_gate is None else calibration_gate.protocol_id
            ),
            "score_version": distribution.score_version,
            "source_score_collection_id": (distribution.source_score_collection_id),
            "candidate_edge_ids": list(universe.candidate_edge_ids),
            "plan_ids": list(distribution.plan_ids),
            "active_probability_record_ids": [
                item.record_id for item in collection.records
            ],
            "local_fdr_diagnostic_ids": [
                item.diagnostic_id for item in collection.diagnostics
            ],
            "status_counts": [list(item) for item in collection.status_counts],
            "universe": universe.to_dict(),
            "active_probability_spec": spec.to_dict(),
            "calibration_gate": (
                None if calibration_gate is None else calibration_gate.to_dict()
            ),
            "tables": records,
        }
        manifest["artifact_id"] = stable_id(
            "active_probability_result",
            manifest,
            schema_version=_ARTIFACT_ID_SCHEMA_VERSION,
        )
        _validate_manifest(cast(dict[str, Any], manifest))
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "artifact_schema_version": (ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION),
                "artifact_kind": ACTIVE_PROBABILITY_ARTIFACT_KIND,
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
            "Active-probability result write failed and was marked incomplete",
            code="active_probability_write_failed",
            field="destination",
            remediation="Inspect producer diagnostics and write to a new directory",
        ) from error
    return ActiveProbabilityResult.load(
        output,
        calibration_result_path=(
            None if replayed_calibration is None else replayed_calibration.path
        ),
        replay_registry=replay_registry,
        n_jobs=n_jobs,
    )


_MANIFEST_FIELDS = {
    "artifact_schema_version",
    "artifact_kind",
    "artifact_id",
    "status",
    "release_mode",
    "comm_probability_release_allowed",
    "collection_id",
    "distribution_id",
    "active_null_id",
    "active_null_spec_id",
    "candidate_universe_id",
    "candidate_universe_policy_id",
    "score_spec_id",
    "active_probability_spec_id",
    "calibration_gate_id",
    "calibration_gate_status",
    "calibration_result_artifact_id",
    "calibration_campaign_artifact_id",
    "calibration_evidence_id",
    "calibration_protocol_id",
    "score_version",
    "source_score_collection_id",
    "candidate_edge_ids",
    "plan_ids",
    "active_probability_record_ids",
    "local_fdr_diagnostic_ids",
    "status_counts",
    "universe",
    "active_probability_spec",
    "calibration_gate",
    "tables",
}


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    load_schema_document("active_probability_result.schema.json")
    if set(manifest) != _MANIFEST_FIELDS:
        _fail(
            "Active-probability manifest fields do not match the current schema",
            code="invalid_active_probability_manifest",
            field_name="manifest",
        )
    if (
        manifest["artifact_schema_version"] != ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION
        or manifest["artifact_kind"] != ACTIVE_PROBABILITY_ARTIFACT_KIND
        or manifest["status"] != _COMPLETE
        or type(manifest["comm_probability_release_allowed"]) is not bool
    ):
        _fail(
            "Active-probability manifest metadata is invalid",
            code="invalid_active_probability_manifest",
            field_name="artifact_schema_version,status",
        )
    released = bool(manifest["comm_probability_release_allowed"])
    expected_names = {
        ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE,
        ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE,
        ACTIVE_PROBABILITY_NULL_SOURCE_TABLE,
    }
    if released:
        expected_names.add(ACTIVE_PROBABILITY_TABLE)
    if (
        manifest["release_mode"] != (_RELEASED if released else _DIAGNOSTIC_ONLY)
        or (released and manifest["calibration_gate_status"] != "passed")
        or (released and manifest["calibration_gate_id"] is None)
        or set(cast(dict[str, Any], manifest["tables"])) != expected_names
    ):
        _fail(
            "Active-probability release mode, gate, and tables disagree",
            code="invalid_active_probability_release_manifest",
            field_name="release_mode,calibration_gate_status,tables",
        )
    calibration_ids = (
        manifest["calibration_campaign_artifact_id"],
        manifest["calibration_evidence_id"],
        manifest["calibration_protocol_id"],
    )
    calibration_result_id = manifest["calibration_result_artifact_id"]
    if manifest["calibration_gate_id"] is None:
        if (
            manifest["calibration_gate_status"] is not None
            or manifest["calibration_gate"] is not None
            or calibration_result_id is not None
            or any(item is not None for item in calibration_ids)
        ):
            _fail(
                "Absent G3-P gate has non-null calibration lineage",
                code="invalid_active_probability_release_manifest",
                field_name="calibration_gate_id,calibration_campaign_artifact_id",
            )
    elif any(not isinstance(item, str) or not item for item in calibration_ids):
        _fail(
            "Present G3-P gate lacks complete campaign lineage",
            code="invalid_active_probability_release_manifest",
            field_name="calibration_campaign_artifact_id,calibration_evidence_id",
        )
    if released:
        if not isinstance(calibration_result_id, str) or not calibration_result_id:
            _fail(
                "Released probabilities lack their G3-P result artifact identity",
                code="invalid_active_probability_release_manifest",
                field_name="calibration_result_artifact_id",
            )
    elif calibration_result_id is not None:
        _fail(
            "Diagnostic-only probabilities cannot claim a verified G3-P result",
            code="invalid_active_probability_release_manifest",
            field_name="calibration_result_artifact_id",
        )
    for field_name in (
        "artifact_id",
        "collection_id",
        "distribution_id",
        "active_null_id",
        "active_null_spec_id",
        "candidate_universe_id",
        "candidate_universe_policy_id",
        "score_spec_id",
        "active_probability_spec_id",
        "score_version",
        "source_score_collection_id",
    ):
        value = manifest[field_name]
        if not isinstance(value, str) or not value:
            _fail(
                "Active-probability manifest has an invalid identifier",
                code="invalid_active_probability_manifest",
                field_name=field_name,
            )
    if manifest["artifact_id"] != stable_id(
        "active_probability_result",
        _manifest_payload(manifest),
        schema_version=_ARTIFACT_ID_SCHEMA_VERSION,
    ):
        _fail(
            "Active-probability artifact identity is inconsistent",
            code="active_probability_artifact_identity_mismatch",
            field_name="artifact_id",
        )
    if manifest["collection_id"] != _collection_id_from_manifest(manifest):
        _fail(
            "Active-probability collection identity is inconsistent",
            code="active_probability_collection_identity_mismatch",
            field_name="collection_id",
        )
    edge_ids = manifest["candidate_edge_ids"]
    plan_ids = manifest["plan_ids"]
    record_ids = manifest["active_probability_record_ids"]
    diagnostic_ids = manifest["local_fdr_diagnostic_ids"]
    if (
        not isinstance(edge_ids, list)
        or not edge_ids
        or edge_ids != sorted(set(edge_ids))
        or not isinstance(plan_ids, list)
        or not plan_ids
        or plan_ids != sorted(set(plan_ids))
        or not isinstance(record_ids, list)
        or len(record_ids) != len(edge_ids)
        or len(set(record_ids)) != len(record_ids)
        or not isinstance(diagnostic_ids, list)
        or not diagnostic_ids
        or len(set(diagnostic_ids)) != len(diagnostic_ids)
    ):
        _fail(
            "Active-probability manifest registries are incomplete or duplicated",
            code="invalid_active_probability_registry",
            field_name="candidate_edge_ids,plan_ids,record_ids,diagnostic_ids",
        )
    raw_tables = cast(dict[str, Any], manifest["tables"])
    for name in expected_names:
        raw = raw_tables[name]
        contract = _table_contract(name)
        if not isinstance(raw, dict) or set(raw) != {
            "filename",
            "rows",
            "sha256",
            "schema",
        }:
            _fail(
                "Active-probability table record is invalid",
                code="invalid_active_probability_manifest",
                field_name=name,
            )
        if (
            raw["filename"] != contract.filename
            or raw["schema"] != contract.schema_filename
            or isinstance(raw["rows"], bool)
            or not isinstance(raw["rows"], int)
            or raw["rows"] < 1
            or not isinstance(raw["sha256"], str)
            or len(raw["sha256"]) != 64
        ):
            _fail(
                "Active-probability table record metadata is invalid",
                code="invalid_active_probability_manifest",
                field_name=name,
            )
    return manifest


def _universe_from_manifest(manifest: Mapping[str, Any]) -> FrozenActiveEdgeUniverse:
    value = manifest["universe"]
    if not isinstance(value, Mapping):
        _fail(
            "Active-probability universe registry is invalid",
            code="invalid_active_probability_registry",
            field_name="universe",
        )
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list):
        _fail(
            "Active-probability candidate registry is invalid",
            code="invalid_active_probability_registry",
            field_name="universe.candidates",
        )
    try:
        candidates = tuple(
            ActiveEdgeCandidate(
                contrast_id=str(item["contrast_id"]),
                context_id=str(item["context_id"]),
                sender=str(item["sender"]),
                receiver=str(item["receiver"]),
                interaction_id=str(item["interaction_id"]),
                driver_id=str(item["driver_id"]),
                mode=str(item["mode"]),
            )
            for item in raw_candidates
            if isinstance(item, Mapping)
        )
        if len(candidates) != len(raw_candidates):
            raise ValueError("candidate rows are invalid")
        universe = freeze_active_edge_universe(
            candidates,
            universe_name=str(value["universe_name"]),
            contrast_id=str(value["contrast_id"]),
            score_version=str(value["score_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Active-probability universe registry cannot be reconstructed",
            code="invalid_active_probability_registry",
            field="universe",
            remediation="Reject the artifact and regenerate it",
        ) from error
    if universe.to_dict() != dict(value):
        _fail(
            "Active-probability universe registry identity is inconsistent",
            code="active_probability_universe_mismatch",
            field_name="candidate_universe_id",
        )
    return universe


def _spec_from_manifest(manifest: Mapping[str, Any]) -> ActiveProbabilitySpec:
    value = manifest["active_probability_spec"]
    if not isinstance(value, Mapping):
        _fail(
            "Active-probability specification registry is invalid",
            code="invalid_active_probability_registry",
            field_name="active_probability_spec",
        )
    try:
        spec = ActiveProbabilitySpec(
            minimum_candidate_edges=int(value["minimum_candidate_edges"]),
            minimum_null_plans=int(value["minimum_null_plans"]),
            estimator=str(value["estimator"]),
            stratum_policy=str(value["stratum_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Active-probability specification cannot be reconstructed",
            code="invalid_active_probability_registry",
            field="active_probability_spec",
            remediation="Reject the artifact and regenerate it",
        ) from error
    if spec.to_dict() != dict(value):
        _fail(
            "Active-probability specification identity is inconsistent",
            code="active_probability_spec_mismatch",
            field_name="active_probability_spec_id",
        )
    return spec


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
            f"Active-probability table {name!r} is missing or corrupted",
            code="corrupted_active_probability_table",
            field=name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if digest != record["sha256"]:
        _fail(
            f"Active-probability table {name!r} does not match its manifest",
            code="active_probability_digest_mismatch",
            field_name=name,
        )
    try:
        frame = _read_parquet(path, engine="pyarrow")
    except Exception as error:
        raise ResultValidationError(
            f"Active-probability table {name!r} is missing or corrupted",
            code="corrupted_active_probability_table",
            field=name,
            remediation="Reject the artifact and regenerate it",
        ) from error
    if len(frame) != record["rows"]:
        _fail(
            f"Active-probability table {name!r} does not match its manifest",
            code="active_probability_digest_mismatch",
            field_name=name,
        )
    return _validate_table(name, frame)


def _point_records_from_diagnostics(
    frame: pd.DataFrame,
) -> tuple[PointActiveEdgeScoreRecord, ...]:
    records: list[PointActiveEdgeScoreRecord] = []
    for row in frame.itertuples(index=False):
        source = cast(Any, row)
        records.append(
            PointActiveEdgeScoreRecord(
                candidate_edge_id=str(source.candidate_edge_id),
                stratum_id=str(source.stratum_id),
                score_version=str(source.score_version),
                source_score_collection_id=str(source.source_score_collection_id),
                n_subjects=int(source.n_subjects),
                score=(
                    None
                    if _is_missing(source.point_score)
                    else float(source.point_score)
                ),
                status=ActiveEdgeScoreStatus(str(source.point_status)),
                reason_code=(
                    None
                    if _is_missing(source.point_reason_code)
                    else str(source.point_reason_code)
                ),
            )
        )
    return tuple(records)


def _null_records_from_table(
    frame: pd.DataFrame,
) -> tuple[NullActiveEdgeScoreRecord, ...]:
    records: list[NullActiveEdgeScoreRecord] = []
    for row in frame.itertuples(index=False):
        source = cast(Any, row)
        records.append(
            NullActiveEdgeScoreRecord(
                candidate_edge_id=str(source.candidate_edge_id),
                plan_id=str(source.plan_id),
                stratum_id=str(source.stratum_id),
                score_version=str(source.score_version),
                null_rerun_record_id=str(source.null_rerun_record_id),
                source_score_collection_id=(
                    None
                    if _is_missing(source.source_score_collection_id)
                    else str(source.source_score_collection_id)
                ),
                n_subjects=(
                    None if _is_missing(source.n_subjects) else int(source.n_subjects)
                ),
                score=None if _is_missing(source.score) else float(source.score),
                status=ActiveEdgeScoreStatus(str(source.status)),
                reason_code=(
                    None if _is_missing(source.reason_code) else str(source.reason_code)
                ),
            )
        )
    return tuple(records)


_RECOMPUTED_CANDIDATE_COLUMNS = (
    "candidate_edge_id",
    "point_record_id",
    "source_score_collection_id",
    "stratum_id",
    "score_version",
    "n_subjects",
    "point_score",
    "point_status",
    "point_reason_code",
    "active_null_empirical_p_value",
    "candidate_local_fdr",
    "candidate_comm_probability",
    "local_fdr_diagnostic_id",
)


def _validate_recomputed_probability_values(
    manifest: Mapping[str, Any],
    candidate: pd.DataFrame,
    strata: pd.DataFrame,
    *,
    distribution: NullScoreDistribution,
    spec: ActiveProbabilitySpec,
) -> None:
    try:
        recomputed = estimate_active_probabilities(distribution, spec=spec)
        expected_candidate = _candidate_diagnostics(recomputed, distribution)
        expected_strata = _stratum_diagnostics(recomputed)
        pd.testing.assert_frame_equal(
            candidate[list(_RECOMPUTED_CANDIDATE_COLUMNS)].reset_index(drop=True),
            expected_candidate[list(_RECOMPUTED_CANDIDATE_COLUMNS)].reset_index(
                drop=True
            ),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            strata.reset_index(drop=True),
            expected_strata.reset_index(drop=True),
            check_exact=True,
        )
    except (AssertionError, ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted probabilities differ from a fresh producer recomputation",
            code="active_probability_recomputation_mismatch",
            field="candidate_local_fdr,candidate_comm_probability",
            remediation="Reject the artifact and regenerate it",
        ) from error
    expected_status_counts = [list(item) for item in recomputed.status_counts]
    if manifest["status_counts"] != expected_status_counts:
        _fail(
            "Active-probability status counts differ from recomputed records",
            code="active_probability_recomputation_mismatch",
            field_name="status_counts",
        )
    gate_id = manifest["calibration_gate_id"]
    actual_gate_ids = tuple(
        None if _is_missing(value) else str(value)
        for value in candidate["calibration_gate_id"].array
    )
    if any(value != gate_id for value in actual_gate_ids):
        _fail(
            "Candidate rows do not share the manifest G3-P gate",
            code="active_probability_gate_binding_mismatch",
            field_name="calibration_gate_id",
        )
    released = bool(manifest["comm_probability_release_allowed"])
    recomputed_by_edge = {item.candidate_edge_id: item for item in recomputed.records}
    raw_gate = manifest["calibration_gate"]
    gate_reason = (
        None
        if not isinstance(raw_gate, Mapping) or raw_gate.get("reason_code") is None
        else str(raw_gate["reason_code"])
    )
    reconstructed_ids: list[str] = []
    for row in candidate.itertuples(index=False):
        source = cast(Any, row)
        recomputed_record = recomputed_by_edge[str(source.candidate_edge_id)]
        expected_reason = recomputed_record.probability_reason_code
        if expected_reason == "g3p_gate_not_passed":
            expected_reason = None if released else gate_reason or expected_reason
        persisted_reason = (
            None
            if _is_missing(source.probability_reason_code)
            else str(source.probability_reason_code)
        )
        if persisted_reason != expected_reason:
            _fail(
                "Candidate probability reason differs from recomputed state",
                code="active_probability_recomputation_mismatch",
                field_name="probability_reason_code",
            )
        candidate_probability = (
            None
            if _is_missing(source.candidate_comm_probability)
            else float(source.candidate_comm_probability)
        )
        record = ActiveProbabilityRecord._from_values(
            candidate_edge_id=str(source.candidate_edge_id),
            point_record_id=str(source.point_record_id),
            stratum_id=str(source.stratum_id),
            score_version=str(source.score_version),
            point_score=(
                None if _is_missing(source.point_score) else float(source.point_score)
            ),
            point_status=ActiveEdgeScoreStatus(str(source.point_status)),
            point_reason_code=(
                None
                if _is_missing(source.point_reason_code)
                else str(source.point_reason_code)
            ),
            active_null_empirical_p_value=(
                None
                if _is_missing(source.active_null_empirical_p_value)
                else float(source.active_null_empirical_p_value)
            ),
            candidate_local_fdr=(
                None
                if _is_missing(source.candidate_local_fdr)
                else float(source.candidate_local_fdr)
            ),
            candidate_comm_probability=candidate_probability,
            comm_probability=candidate_probability if released else None,
            probability_reason_code=persisted_reason,
            local_fdr_diagnostic_id=str(source.local_fdr_diagnostic_id),
            calibration_gate_id=gate_id,
        )
        reconstructed_ids.append(record.record_id)
    persisted_ids = tuple(candidate["active_probability_record_id"].astype(str))
    if tuple(reconstructed_ids) != persisted_ids:
        _fail(
            "Persisted active-probability records fail identity reconstruction",
            code="active_probability_record_binding_mismatch",
            field_name="active_probability_record_id",
        )


def _validate_calibration_result_link(
    manifest: Mapping[str, Any],
    calibration_result: G3PCalibrationResult | None,
) -> None:
    released = bool(manifest["comm_probability_release_allowed"])
    if not released:
        if calibration_result is not None:
            _fail(
                "Diagnostic active-probability result has an unexpected G3-P parent",
                code="active_probability_calibration_result_binding_mismatch",
                field_name="calibration_result_artifact_id",
            )
        return
    if calibration_result is None:
        _fail(
            "Released active probabilities were not verified against their G3-P result",
            code="active_probability_calibration_result_unavailable",
            field_name="calibration_result",
        )
    parent = calibration_result.manifest
    parent_gate = calibration_result.calibration_gate.to_dict()
    attestation = parent["generator_attestation"]
    evidence = parent["evidence"]
    if (
        parent["artifact_id"] != manifest["calibration_result_artifact_id"]
        or parent["campaign_id"] != manifest["calibration_campaign_artifact_id"]
        or parent["evidence_id"] != manifest["calibration_evidence_id"]
        or parent["protocol_id"] != manifest["calibration_protocol_id"]
        or parent["gate_id"] != manifest["calibration_gate_id"]
        or parent["gate_status"] != manifest["calibration_gate_status"]
        or parent["comm_probability_release_allowed"] is not True
        or parent["calibration_gate"] != manifest["calibration_gate"]
        or parent_gate != manifest["calibration_gate"]
        or not isinstance(attestation, Mapping)
        or not isinstance(evidence, Mapping)
        or evidence.get("generator_attestation_id") != attestation.get("attestation_id")
        or parent_gate.get("generator_attestation_id")
        != attestation.get("attestation_id")
        or attestation.get("release_approved") is not True
    ):
        _fail(
            "Active-probability calibration lineage differs from replayed G3-P result",
            code="active_probability_calibration_result_binding_mismatch",
            field_name=(
                "calibration_result_artifact_id,calibration_campaign_artifact_id,"
                "calibration_evidence_id,calibration_gate_id"
            ),
        )


def _validate_links(
    manifest: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    *,
    calibration_result: G3PCalibrationResult | None,
) -> None:
    _validate_calibration_result_link(manifest, calibration_result)
    universe = _universe_from_manifest(manifest)
    spec = _spec_from_manifest(manifest)
    candidate = tables[ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE]
    strata = tables[ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE]
    nulls = tables[ACTIVE_PROBABILITY_NULL_SOURCE_TABLE]
    edge_ids = tuple(candidate["candidate_edge_id"].astype(str))
    if (
        edge_ids != universe.candidate_edge_ids
        or edge_ids != tuple(manifest["candidate_edge_ids"])
        or tuple(candidate["active_probability_record_id"].astype(str))
        != tuple(manifest["active_probability_record_ids"])
        or tuple(strata["local_fdr_diagnostic_id"].astype(str))
        != tuple(manifest["local_fdr_diagnostic_ids"])
        or set(candidate["local_fdr_diagnostic_id"].astype(str))
        != set(strata["local_fdr_diagnostic_id"].astype(str))
        or set(candidate["source_score_collection_id"].astype(str))
        != {str(manifest["source_score_collection_id"])}
        or set(candidate["score_version"].astype(str))
        != {str(manifest["score_version"])}
        or manifest["candidate_universe_id"] != universe.universe_id
        or manifest["active_probability_spec_id"] != spec.spec_id
    ):
        _fail(
            "Active-probability candidate, source, universe, or spec links differ",
            code="active_probability_linkage_mismatch",
            field_name="candidate_edge_id,source_score_collection_id",
        )
    point_records = _point_records_from_diagnostics(candidate)
    expected_point_ids = tuple(item.record_id for item in point_records)
    if tuple(candidate["point_record_id"].astype(str)) != expected_point_ids:
        _fail(
            "Persisted point records fail identity reconstruction",
            code="active_probability_point_binding_mismatch",
            field_name="point_record_id",
        )
    null_records = _null_records_from_table(nulls)
    expected_null_ids = tuple(item.record_id for item in null_records)
    if tuple(nulls["null_record_id"].astype(str)) != expected_null_ids:
        _fail(
            "Persisted null records fail identity reconstruction",
            code="active_probability_null_binding_mismatch",
            field_name="null_record_id",
        )
    try:
        distribution = NullScoreDistribution(
            active_null_id=str(manifest["active_null_id"]),
            active_null_spec_id=str(manifest["active_null_spec_id"]),
            candidate_universe_id=str(manifest["candidate_universe_id"]),
            candidate_universe_policy_id=str(manifest["candidate_universe_policy_id"]),
            score_spec_id=str(manifest["score_spec_id"]),
            point_records=point_records,
            plan_ids=tuple(cast(list[str], manifest["plan_ids"])),
            null_records=null_records,
        )
    except (ContractError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Persisted active-null sources do not form the exact distribution",
            code="active_probability_null_rectangle_mismatch",
            field="candidate_edge_id,plan_id",
            remediation="Reject the artifact and regenerate it",
        ) from error
    if distribution.distribution_id != manifest["distribution_id"]:
        _fail(
            "Persisted active-null distribution identity differs",
            code="active_probability_distribution_mismatch",
            field_name="distribution_id",
        )
    _validate_recomputed_probability_values(
        manifest,
        candidate,
        strata,
        distribution=distribution,
        spec=spec,
    )
    gate = manifest["calibration_gate"]
    gate_id = manifest["calibration_gate_id"]
    if gate_id is None:
        if gate is not None or manifest["calibration_gate_status"] is not None:
            _fail(
                "Absent G3-P gate has inconsistent registry fields",
                code="active_probability_gate_binding_mismatch",
                field_name="calibration_gate",
            )
    elif not isinstance(gate, Mapping) or (
        gate.get("gate_id") != gate_id
        or gate.get("status") != manifest["calibration_gate_status"]
        or gate.get("active_null_spec_id") != manifest["active_null_spec_id"]
        or gate.get("active_probability_spec_id")
        != manifest["active_probability_spec_id"]
        or gate.get("score_spec_id") != manifest["score_spec_id"]
        or gate.get("candidate_universe_policy_id")
        != manifest["candidate_universe_policy_id"]
        or gate.get("source_artifact_id")
        != manifest["calibration_campaign_artifact_id"]
        or gate.get("evidence_id") != manifest["calibration_evidence_id"]
        or gate.get("protocol_id") != manifest["calibration_protocol_id"]
        or gate.get("estimator_id") != spec.estimator_id
        or gate.get("stratum_policy_id") != spec.stratum_policy_id
        or gate.get("score_version") != manifest["score_version"]
        or (
            bool(manifest["comm_probability_release_allowed"])
            and (
                gate.get("generator_verified") is not True
                or not isinstance(gate.get("generator_attestation_id"), str)
                or not gate.get("generator_attestation_id")
            )
        )
    ):
        _fail(
            "G3-P gate registry differs from runtime contracts",
            code="active_probability_gate_binding_mismatch",
            field_name="calibration_gate",
        )
    if gate_id is not None:
        if not isinstance(gate, Mapping):
            _fail(
                "G3-P gate registry is unavailable for a bound gate",
                code="active_probability_gate_binding_mismatch",
                field_name="calibration_gate",
            )
        identity_payload = {
            str(key): value
            for key, value in gate.items()
            if key
            not in {
                "gate_id",
                "comm_probability_release_allowed",
                "thresholds",
            }
        }
        identity_payload["producer_marker"] = _GATE_PRODUCER
        if gate_id != stable_id(
            "g3p_calibration_gate",
            identity_payload,
            schema_version="3",
        ):
            _fail(
                "Persisted G3-P gate identity cannot be reconstructed",
                code="active_probability_gate_identity_mismatch",
                field_name="calibration_gate_id",
            )
    released = bool(manifest["comm_probability_release_allowed"])
    if released:
        public = tables[ACTIVE_PROBABILITY_TABLE]
        if tuple(public["candidate_edge_id"].astype(str)) != edge_ids:
            _fail(
                "Public probabilities do not cover the exact candidate universe",
                code="active_probability_public_coverage_mismatch",
                field_name="candidate_edge_id",
            )
        public_by_edge = public.set_index("candidate_edge_id")
        universe_by_edge = {
            item.candidate_edge_id: item for item in universe.candidates
        }
        for row in candidate.itertuples(index=False):
            source = cast(Any, row)
            candidate_id = str(source.candidate_edge_id)
            public_row = cast(Any, public_by_edge.loc[candidate_id])
            universe_candidate = universe_by_edge[candidate_id]
            candidate_value = source.candidate_comm_probability
            public_value = public_row["comm_probability"]
            if _is_missing(candidate_value) != _is_missing(public_value) or (
                not _is_missing(candidate_value)
                and not math.isclose(
                    float(candidate_value), float(public_value), abs_tol=1e-12
                )
            ):
                _fail(
                    "Public probability differs from its gated candidate value",
                    code="active_probability_public_projection_mismatch",
                    field_name="comm_probability",
                )
            expected_reason = (
                None
                if not _is_missing(candidate_value)
                else (
                    None
                    if _is_missing(source.probability_reason_code)
                    else str(source.probability_reason_code)
                )
            )
            actual_reason = (
                None
                if _is_missing(public_row["reason_code"])
                else str(public_row["reason_code"])
            )
            expected_labels = {
                "contrast_id": universe_candidate.contrast_id,
                "context_id": universe_candidate.context_id,
                "sender": universe_candidate.sender,
                "receiver": universe_candidate.receiver,
                "interaction_id": universe_candidate.interaction_id,
                "driver_id": universe_candidate.driver_id,
                "mode": CommunicationMode(universe_candidate.mode).value,
                "stratum_id": str(source.stratum_id),
                "score_version": str(source.score_version),
                "status": (
                    _RELEASED if not _is_missing(candidate_value) else "not_estimable"
                ),
                "active_probability_record_id": str(
                    source.active_probability_record_id
                ),
            }
            if (
                any(
                    str(public_row[field_name]) != expected
                    for field_name, expected in expected_labels.items()
                )
                or actual_reason != expected_reason
            ):
                _fail(
                    "Public probability labels differ from the frozen universe",
                    code="active_probability_public_projection_mismatch",
                    field_name="sender,receiver,interaction_id,status,reason_code",
                )


@dataclass(frozen=True, slots=True, init=False)
class ActiveProbabilityResult:
    """Immutable facade over one independent ADR-013 result directory."""

    path: Path
    _manifest: dict[str, Any] = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActiveProbabilityResult is producer-owned; use "
            "write_active_probability_result() or load()"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        calibration_result_path: str | Path | None = None,
        replay_registry: CalibrationReplayRegistry | None = None,
        n_jobs: int = 1,
    ) -> ActiveProbabilityResult:
        """Load one artifact, replaying its G3-P parent before any release."""

        root = Path(path)
        try:
            marker = _read_json(root / _STATUS_FILENAME)
        except Exception as error:
            raise ResultValidationError(
                "Active-probability status marker is missing or corrupted",
                code="invalid_active_probability_status",
                field="path",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if marker.get("status") != _COMPLETE:
            raise IncompleteResultError(
                "Active-probability result is incomplete",
                code="incomplete_active_probability_result",
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
            != ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION
            or marker.get("artifact_kind") != ACTIVE_PROBABILITY_ARTIFACT_KIND
        ):
            _fail(
                "Active-probability status marker violates the current schema",
                code="invalid_active_probability_status",
                field_name="path",
            )
        try:
            manifest = _validate_manifest(_read_json(root / _MANIFEST_FILENAME))
        except ResultValidationError:
            raise
        except Exception as error:
            raise ResultValidationError(
                "Active-probability manifest is missing or corrupted",
                code="invalid_active_probability_manifest",
                field="path",
                remediation="Reject the artifact and regenerate it",
            ) from error
        if marker["artifact_id"] != manifest["artifact_id"]:
            _fail(
                "Active-probability status and manifest identities differ",
                code="active_probability_status_manifest_mismatch",
                field_name="artifact_id",
            )
        calibration_result: G3PCalibrationResult | None = None
        if bool(manifest["comm_probability_release_allowed"]):
            if calibration_result_path is None:
                _fail(
                    "Released probabilities require their G3-P result directory",
                    code="active_probability_calibration_result_unavailable",
                    field_name="calibration_result_path",
                    remediation=(
                        "Load with the exact G3-P result directory and replay registry"
                    ),
                )
            if replay_registry is None:
                _fail(
                    "Released probabilities require their G3-P replay registry",
                    code="active_probability_calibration_replay_unavailable",
                    field_name="replay_registry",
                    remediation=(
                        "Load with the exact registry used to replay the G3-P ledger"
                    ),
                )
            calibration_result = G3PCalibrationResult.load(
                calibration_result_path,
                replay_registry=replay_registry,
                n_jobs=n_jobs,
            )
        raw_records = cast(dict[str, Mapping[str, Any]], manifest["tables"])
        tables = {
            name: _load_table(root, name, record)
            for name, record in raw_records.items()
        }
        _validate_links(
            manifest,
            tables,
            calibration_result=calibration_result,
        )
        self = object.__new__(cls)
        object.__setattr__(self, "path", root.resolve())
        object.__setattr__(self, "_manifest", copy.deepcopy(manifest))
        return self

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive copy of the complete source registry."""

        return copy.deepcopy(self._manifest)

    @property
    def has_released_probabilities(self) -> bool:
        """Whether a passed bound G3-P gate authorized the public table."""

        return bool(self._manifest["comm_probability_release_allowed"])

    @property
    def source_registry(self) -> dict[str, Any]:
        """Return the immutable-by-copy parent and null registry."""

        fields = {
            key: self._manifest[key]
            for key in (
                "collection_id",
                "distribution_id",
                "active_null_id",
                "active_null_spec_id",
                "candidate_universe_id",
                "candidate_universe_policy_id",
                "score_spec_id",
                "active_probability_spec_id",
                "calibration_gate_id",
                "calibration_gate_status",
                "calibration_result_artifact_id",
                "calibration_campaign_artifact_id",
                "calibration_evidence_id",
                "calibration_protocol_id",
                "score_version",
                "source_score_collection_id",
                "candidate_edge_ids",
                "plan_ids",
                "universe",
                "active_probability_spec",
                "calibration_gate",
            )
        }
        return copy.deepcopy(fields)

    def _read(
        self,
        name: str,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        raw_records = cast(dict[str, Mapping[str, Any]], self._manifest["tables"])
        if name not in raw_records:
            raise KeyError(name)
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

    def read_active_probabilities(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read only G3-P-released public probabilities."""

        if not self.has_released_probabilities:
            raise KeyError(ACTIVE_PROBABILITY_TABLE)
        return self._read(
            ACTIVE_PROBABILITY_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_candidate_diagnostics(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read explicitly diagnostic active-null p and candidate local-FDR values."""

        return self._read(
            ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_stratum_diagnostics(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read BUM fit diagnostics without recomputing the estimator."""

        return self._read(
            ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE,
            filters=filters,
            columns=columns,
        )

    def read_null_sources(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read the complete candidate by active-null-plan source rectangle."""

        return self._read(
            ACTIVE_PROBABILITY_NULL_SOURCE_TABLE,
            filters=filters,
            columns=columns,
        )


__all__ = [
    "ACTIVE_PROBABILITY_ARTIFACT_KIND",
    "ACTIVE_PROBABILITY_CANDIDATE_DIAGNOSTIC_TABLE",
    "ACTIVE_PROBABILITY_NULL_SOURCE_TABLE",
    "ACTIVE_PROBABILITY_RESULT_SCHEMA_VERSION",
    "ACTIVE_PROBABILITY_STRATUM_DIAGNOSTIC_TABLE",
    "ACTIVE_PROBABILITY_TABLE",
    "ActiveProbabilityResult",
    "write_active_probability_result",
]
