"""Atomic persistence for the v2 sample-level edge score artifact."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pandas as pd

from crychic.core import canonical_json, stable_id
from crychic.scoring import (
    SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION,
    SampleEdgeScoreV2,
    SampleEdgeScoreV2Provenance,
)

from .errors import IncompleteResultError, ResultValidationError, ResultWriteError

SAMPLE_EDGE_SCORE_V2_ARTIFACT_KIND = "crychic_sample_edge_score_v2"
SAMPLE_EDGE_SCORE_V2_FILENAME = "sample_edge_scores_v2.parquet"
SAMPLE_EDGE_SCORE_V2_MANIFEST_FILENAME = "sample_edge_scores_v2_manifest.json"
_SCHEMA_FILENAME = "sample_edge_scores_v2.schema.json"
_STATUS_FILENAME = "_status.json"


def _read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
    reader = cast(Callable[..., pd.DataFrame], pd.read_parquet)
    return reader(*args, **kwargs)


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
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
    status: dict[str, object] = {
        "artifact_kind": SAMPLE_EDGE_SCORE_V2_ARTIFACT_KIND,
        "artifact_schema_version": SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION,
        "status": "incomplete",
    }
    if error_type is not None:
        status["error_type"] = error_type
    try:
        _write_json(path / _STATUS_FILENAME, status)
    except OSError:
        pass


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _manifest_payload(
    scores: SampleEdgeScoreV2,
    *,
    table_sha256: str,
) -> dict[str, object]:
    return {
        "artifact_kind": SAMPLE_EDGE_SCORE_V2_ARTIFACT_KIND,
        "artifact_schema_version": SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION,
        "status": "complete",
        "provenance": scores.provenance.to_dict(),
        "table": {
            "filename": SAMPLE_EDGE_SCORE_V2_FILENAME,
            "rows": len(scores.table),
            "sha256": table_sha256,
            "schema": _SCHEMA_FILENAME,
        },
    }


@dataclass(frozen=True, slots=True)
class SampleEdgeScoreV2Artifact:
    """Verified immutable view of a persisted v2 sample-edge directory."""

    path: Path
    scores: SampleEdgeScoreV2
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> SampleEdgeScoreV2Artifact:
        """Load a complete artifact while verifying identity and table checksum."""

        root = Path(path).resolve()
        try:
            status = _read_json(root / _STATUS_FILENAME)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ResultValidationError(
                "Sample-edge artifact status is missing or invalid",
                code="invalid_sample_edge_status",
                field="_status.json",
                remediation="Regenerate the sample-edge artifact",
            ) from error
        if status.get("status") != "complete":
            raise IncompleteResultError(
                "Sample-edge artifact is incomplete",
                code="incomplete_sample_edge_artifact",
                field="status",
                remediation="Resume or rerun the producing workflow",
            )
        try:
            manifest = _read_json(root / SAMPLE_EDGE_SCORE_V2_MANIFEST_FILENAME)
            expected_manifest_fields = {
                "artifact_id",
                "artifact_kind",
                "artifact_schema_version",
                "status",
                "provenance",
                "table",
            }
            if set(manifest) != expected_manifest_fields:
                raise ValueError("manifest fields do not match the v2 contract")
            if (
                manifest["artifact_kind"] != SAMPLE_EDGE_SCORE_V2_ARTIFACT_KIND
                or manifest["artifact_schema_version"]
                != SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION
                or manifest["status"] != "complete"
            ):
                raise ValueError("manifest identity is incompatible")
            table_record = manifest["table"]
            if not isinstance(table_record, dict) or table_record.get("filename") != (
                SAMPLE_EDGE_SCORE_V2_FILENAME
            ):
                raise ValueError("manifest table record is incompatible")
            if table_record.get("schema") != _SCHEMA_FILENAME:
                raise ValueError("manifest table schema is incompatible")
            provenance = SampleEdgeScoreV2Provenance.from_dict(manifest["provenance"])
            table_path = root / SAMPLE_EDGE_SCORE_V2_FILENAME
            if _sha256(table_path) != table_record.get("sha256"):
                raise ValueError(
                    "sample-edge table checksum does not match its manifest"
                )
            table = _read_parquet(table_path, engine="pyarrow")
            if len(table) != table_record.get("rows"):
                raise ValueError("sample-edge row count does not match its manifest")
            scores = SampleEdgeScoreV2(table=table, provenance=provenance)
            payload = {
                key: value
                for key, value in manifest.items()
                if key != "artifact_id"
            }
            expected_id = stable_id(
                "sample_edge_score_v2_artifact",
                payload,
                schema_version=SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION,
            )
            if manifest["artifact_id"] != expected_id:
                raise ValueError("sample-edge artifact ID does not match its payload")
            if status.get("artifact_id") != expected_id:
                raise ValueError("sample-edge status ID does not match the manifest")
        except IncompleteResultError:
            raise
        except Exception as error:
            raise ResultValidationError(
                "Sample-edge artifact is missing or corrupted",
                code="invalid_sample_edge_artifact",
                field=str(root),
                remediation="Reject the directory and regenerate it from intact inputs",
            ) from error
        return cls(
            path=root,
            scores=scores,
            manifest=_freeze(copy.deepcopy(manifest)),
        )


def write_sample_edge_score_v2(
    destination: str | Path,
    scores: SampleEdgeScoreV2,
) -> SampleEdgeScoreV2Artifact:
    """Validate and atomically publish one fold-level sample-edge artifact."""

    if not isinstance(scores, SampleEdgeScoreV2):
        raise TypeError("scores must be a SampleEdgeScoreV2")
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"Sample-edge destination {output.name!r} already exists",
            code="sample_edge_destination_exists",
            field="destination",
            remediation="Choose a new versioned artifact directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        table_path = temporary / SAMPLE_EDGE_SCORE_V2_FILENAME
        scores.table.to_parquet(
            table_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        payload = _manifest_payload(scores, table_sha256=_sha256(table_path))
        artifact_id = stable_id(
            "sample_edge_score_v2_artifact",
            payload,
            schema_version=SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION,
        )
        _write_json(
            temporary / SAMPLE_EDGE_SCORE_V2_MANIFEST_FILENAME,
            {"artifact_id": artifact_id, **payload},
        )
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "artifact_id": artifact_id,
                "artifact_kind": SAMPLE_EDGE_SCORE_V2_ARTIFACT_KIND,
                "artifact_schema_version": SAMPLE_EDGE_SCORE_V2_SCHEMA_VERSION,
                "status": "complete",
            },
        )
        os.replace(temporary, output)
    except Exception as error:
        if temporary.exists():
            _mark_incomplete(temporary, type(error).__name__)
            if not output.exists():
                os.replace(temporary, output)
        raise ResultWriteError(
            "Sample-edge artifact write failed and was marked incomplete",
            code="sample_edge_write_failed",
            field="destination",
            remediation="Inspect the cause and rerun to a new destination",
        ) from error
    return SampleEdgeScoreV2Artifact.load(output)


__all__ = [
    "SAMPLE_EDGE_SCORE_V2_ARTIFACT_KIND",
    "SAMPLE_EDGE_SCORE_V2_FILENAME",
    "SAMPLE_EDGE_SCORE_V2_MANIFEST_FILENAME",
    "SampleEdgeScoreV2Artifact",
    "write_sample_edge_score_v2",
]
