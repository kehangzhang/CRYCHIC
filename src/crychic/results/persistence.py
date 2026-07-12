"""Atomic persistence for versioned CRYCHIC result directories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from crychic.core import (
    CrychicConfig,
    RunProvenance,
    canonical_digest,
    canonical_json,
)

from ._schema import (
    RESULT_SCHEMA_VERSION,
    TABLE_NAMES,
    edge_evidence_contract,
    table_contract,
    validate_edge_evidence,
    validate_edge_evidence_links,
    validate_table,
)
from .errors import ResultWriteError

if TYPE_CHECKING:
    from .facade import CrychicResult

STATUS_FILENAME = "_status.json"
CONFIG_FILENAME = "config.json"
PROVENANCE_FILENAME = "provenance.json"
MANIFEST_FILENAME = "run_manifest.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a persisted artifact without materializing it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write canonical JSON inside an unpublished temporary directory."""

    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a result-specific error at the caller boundary."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _normalise_config(value: CrychicConfig | Mapping[str, object]) -> CrychicConfig:
    if isinstance(value, CrychicConfig):
        return value
    return CrychicConfig.from_dict(value)


def _normalise_provenance(
    value: RunProvenance | Mapping[str, object],
) -> RunProvenance:
    if isinstance(value, RunProvenance):
        return value
    return RunProvenance.from_dict(value)


def _normalise_stage(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("run manifest stages must be objects")
    expected = {"name", "status", "reason_code"}
    if set(value) != expected:
        raise ValueError("run manifest stage fields do not match the v0.1 schema")
    name = value["name"]
    status = value["status"]
    reason_code = value["reason_code"]
    if not isinstance(name, str) or not name:
        raise ValueError("run manifest stage name must be non-empty")
    if status not in {"complete", "skipped", "failed", "not_estimable"}:
        raise ValueError("run manifest stage has an unsupported status")
    if reason_code is not None and (
        not isinstance(reason_code, str) or not reason_code
    ):
        raise ValueError("run manifest stage reason_code must be null or non-empty")
    if status != "complete" and reason_code is None:
        raise ValueError("non-complete run stages require reason_code")
    return {"name": name, "status": status, "reason_code": reason_code}


def _build_manifest(
    supplied: Mapping[str, object],
    *,
    config: CrychicConfig,
    provenance: RunProvenance,
    table_records: Mapping[str, object],
    extension_records: Mapping[str, object],
) -> dict[str, object]:
    allowed = {
        "run_id",
        "mode",
        "created_at",
        "stages",
        "warnings",
        "workflow_parameters",
    }
    unknown = set(supplied).difference(allowed)
    if unknown:
        raise ValueError(f"unknown run manifest field: {sorted(unknown)[0]}")
    run_id = supplied.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run manifest requires a non-empty run_id")
    mode = supplied.get("mode", "exploratory")
    if mode != "exploratory":
        raise ValueError("v0.1 run manifests must use exploratory mode")
    created_at = supplied.get("created_at", provenance.created_at)
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("run manifest created_at must be a timestamp string")
    raw_stages = supplied.get("stages", ())
    raw_warnings = supplied.get("warnings", ())
    workflow_parameters = supplied.get("workflow_parameters", {})
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, str):
        raise TypeError("run manifest stages must be a sequence")
    if not isinstance(raw_warnings, Sequence) or isinstance(raw_warnings, str):
        raise TypeError("run manifest warnings must be a sequence")
    if not isinstance(workflow_parameters, Mapping):
        raise TypeError("run manifest workflow_parameters must be an object")
    canonical_json(workflow_parameters)
    warnings = tuple(raw_warnings)
    if any(not isinstance(item, str) or not item for item in warnings):
        raise ValueError("run manifest warnings must be non-empty strings")
    manifest: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "mode": "exploratory",
        "created_at": created_at,
        "code_version": provenance.package_version,
        "config_digest": config.digest,
        "workflow_parameters": dict(workflow_parameters),
        "workflow_digest": canonical_digest(workflow_parameters),
        "provenance_digest": provenance.digest,
        "input_digest": provenance.input_digest,
        "resource_digests": dict(provenance.resource_digests),
        "tables": dict(table_records),
        "stages": [_normalise_stage(stage) for stage in raw_stages],
        "warnings": list(warnings),
    }
    if extension_records:
        manifest["extensions"] = dict(extension_records)
    return manifest


def _mark_incomplete(path: Path, error_type: str | None = None) -> None:
    marker: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "incomplete",
    }
    if error_type is not None:
        marker["error_type"] = error_type
    try:
        write_json(path / STATUS_FILENAME, marker)
    except OSError:
        pass


def write_result(
    destination: str | Path,
    *,
    config: CrychicConfig | Mapping[str, object],
    provenance: RunProvenance | Mapping[str, object],
    run_manifest: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    edge_evidence: pd.DataFrame | None = None,
) -> CrychicResult:
    """Validate and atomically publish a complete v0.1 result directory.

    ``edge_evidence`` is an optional, independently versioned result extension.
    Omitting it preserves the original v0.1.0 manifest and directory shape.
    """

    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"Result destination {output.name!r} already exists",
            code="result_destination_exists",
            field="destination",
            remediation="Choose a new versioned result directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        resolved_config = _normalise_config(config)
        resolved_provenance = _normalise_provenance(provenance)
        if resolved_provenance.config_digest != resolved_config.digest:
            raise ValueError("provenance config_digest does not match config")
        if resolved_provenance.result_schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                "provenance result_schema_version must match the v0.1 result schema"
            )
        if set(tables) != set(TABLE_NAMES):
            raise ValueError("result tables do not match the required v0.1 table set")

        write_json(temporary / CONFIG_FILENAME, resolved_config.to_dict())
        write_json(temporary / PROVENANCE_FILENAME, resolved_provenance.to_dict())
        table_records: dict[str, dict[str, object]] = {}
        for name in TABLE_NAMES:
            frame = tables[name]
            validate_table(name, frame)
            contract = table_contract(name)
            table_path = temporary / contract.filename
            frame.to_parquet(table_path, index=False, engine="pyarrow")
            table_records[name] = {
                "filename": contract.filename,
                "rows": len(frame),
                "sha256": sha256_file(table_path),
                "schema": contract.schema_filename,
            }

        extension_records: dict[str, object] = {}
        if edge_evidence is not None:
            extension_contract = edge_evidence_contract()
            validated_edge_evidence = validate_edge_evidence(edge_evidence)
            validate_edge_evidence_links(
                validated_edge_evidence, tables["sample_scores"]
            )
            extension_path = temporary / extension_contract.filename
            validated_edge_evidence.to_parquet(
                extension_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                row_group_size=131_072,
            )
            extension_records[extension_contract.name] = {
                "extension_schema_version": (
                    extension_contract.extension_schema_version
                ),
                "filename": extension_contract.filename,
                "rows": len(validated_edge_evidence),
                "sha256": sha256_file(extension_path),
                "schema": extension_contract.schema_filename,
                "linked_tables": {
                    table_name: table_records[table_name]["sha256"]
                    for table_name in extension_contract.linked_tables
                },
            }

        manifest = _build_manifest(
            run_manifest,
            config=resolved_config,
            provenance=resolved_provenance,
            table_records=table_records,
            extension_records=extension_records,
        )
        write_json(temporary / MANIFEST_FILENAME, manifest)
        write_json(
            temporary / STATUS_FILENAME,
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "status": "complete",
            },
        )
        os.replace(temporary, output)
    except Exception as exc:
        if temporary.exists():
            _mark_incomplete(temporary, type(exc).__name__)
            if not output.exists():
                os.replace(temporary, output)
        raise ResultWriteError(
            f"Result write for {output.name!r} failed and was marked incomplete",
            code="result_write_failed",
            field="destination",
            remediation="Inspect producer diagnostics and write to a new directory",
        ) from exc

    from .facade import CrychicResult

    return CrychicResult.load(output)
