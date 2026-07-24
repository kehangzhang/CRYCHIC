"""Atomic multi-fold persistence for v7 sample-level edge scores."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from crychic.core import canonical_json, stable_id
from crychic.results import (
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
    SampleEdgeScoreV2Artifact,
    write_sample_edge_score_v2,
)

from .crossfit import CrossFitArtifacts

CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION = "1.0.0"
CROSSFIT_SAMPLE_EDGE_V2_ARTIFACT_KIND = "crychic_crossfit_sample_edge_v2"
_MANIFEST_FILENAME = "crossfit_sample_edge_v2_manifest.json"
_STATUS_FILENAME = "_status.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mark_incomplete(path: Path, error_type: str | None = None) -> None:
    value: dict[str, object] = {
        "artifact_kind": CROSSFIT_SAMPLE_EDGE_V2_ARTIFACT_KIND,
        "artifact_schema_version": CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION,
        "status": "incomplete",
    }
    if error_type is not None:
        value["error_type"] = error_type
    try:
        _write_json(path / _STATUS_FILENAME, value)
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class CrossFitSampleEdgeV2Result:
    """Verified multi-fold sample-edge artifact."""

    path: Path
    manifest: Mapping[str, Any]
    fold_artifacts: tuple[SampleEdgeScoreV2Artifact, ...]

    @classmethod
    def load(cls, path: str | Path) -> CrossFitSampleEdgeV2Result:
        """Load and verify every child artifact and its parent linkage."""

        root = Path(path).resolve()
        try:
            status = _read_json(root / _STATUS_FILENAME)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ResultValidationError(
                "Cross-fit sample-edge status is missing or invalid",
                code="invalid_crossfit_sample_edge_status",
                field="_status.json",
                remediation="Regenerate the cross-fit sample-edge artifact",
            ) from error
        if status.get("status") != "complete":
            raise IncompleteResultError(
                "Cross-fit sample-edge artifact is incomplete",
                code="incomplete_crossfit_sample_edge_artifact",
                field="status",
                remediation="Rerun the v7 cross-fit persistence stage",
            )
        try:
            manifest = _read_json(root / _MANIFEST_FILENAME)
            expected = {
                "artifact_id",
                "artifact_kind",
                "artifact_schema_version",
                "status",
                "crossfit_id",
                "spec_id",
                "repeat_id",
                "folds",
            }
            if set(manifest) != expected:
                raise ValueError("cross-fit sample-edge manifest fields are invalid")
            if (
                manifest["artifact_kind"] != CROSSFIT_SAMPLE_EDGE_V2_ARTIFACT_KIND
                or manifest["artifact_schema_version"]
                != CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION
                or manifest["status"] != "complete"
            ):
                raise ValueError("cross-fit sample-edge manifest is incompatible")
            raw_folds = manifest["folds"]
            if not isinstance(raw_folds, list) or not raw_folds:
                raise ValueError("cross-fit sample-edge folds must be non-empty")
            children: list[SampleEdgeScoreV2Artifact] = []
            fold_ids: list[str] = []
            for index, record in enumerate(raw_folds):
                if not isinstance(record, dict) or set(record) != {
                    "directory",
                    "fold_id",
                    "rows",
                    "transform_manifest_id",
                    "provenance_id",
                    "sample_edge_artifact_id",
                }:
                    raise ValueError("cross-fit sample-edge fold record is invalid")
                expected_directory = f"fold-{index:03d}"
                if record["directory"] != expected_directory:
                    raise ValueError("cross-fit sample-edge fold directory is invalid")
                child = SampleEdgeScoreV2Artifact.load(root / expected_directory)
                if (
                    child.scores.provenance.fold_id != record["fold_id"]
                    or child.scores.provenance.provenance_id
                    != record["provenance_id"]
                    or child.scores.provenance.transform_manifest_id
                    != record["transform_manifest_id"]
                    or child.manifest["artifact_id"]
                    != record["sample_edge_artifact_id"]
                    or len(child.scores.table) != record["rows"]
                    or child.scores.provenance.repeat_id != manifest["repeat_id"]
                ):
                    raise ValueError("cross-fit sample-edge child linkage is invalid")
                children.append(child)
                fold_ids.append(str(record["fold_id"]))
            if fold_ids != sorted(fold_ids) or len(fold_ids) != len(set(fold_ids)):
                raise ValueError("cross-fit sample-edge folds are not canonical")
            payload = {
                key: value
                for key, value in manifest.items()
                if key != "artifact_id"
            }
            expected_id = stable_id(
                "crossfit_sample_edge_v2_artifact",
                payload,
                schema_version=CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION,
            )
            if manifest["artifact_id"] != expected_id:
                raise ValueError("cross-fit sample-edge artifact ID is invalid")
            if status.get("artifact_id") != expected_id:
                raise ValueError("cross-fit sample-edge status ID is invalid")
        except IncompleteResultError:
            raise
        except Exception as error:
            raise ResultValidationError(
                "Cross-fit sample-edge artifact is missing or corrupted",
                code="invalid_crossfit_sample_edge_artifact",
                field=str(root),
                remediation="Reject the directory and rerun v7 cross-fit persistence",
            ) from error
        return cls(
            path=root,
            manifest=_freeze(copy.deepcopy(manifest)),
            fold_artifacts=tuple(children),
        )


def write_crossfit_sample_edge_v2_result(
    artifacts: CrossFitArtifacts,
    destination: str | Path,
) -> CrossFitSampleEdgeV2Result:
    """Atomically persist all configured M0 v2 fold outputs."""

    if type(artifacts) is not CrossFitArtifacts:
        raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
    artifacts._require_intact()
    if artifacts.spec.absolute_activity_v2_spec is None:
        raise ResultWriteError(
            "Cross-fit did not configure absolute_activity_v2_spec",
            code="sample_edge_v2_not_configured",
            field="absolute_activity_v2_spec",
            remediation="Rerun cross-fit with a pre-registered M0 v2 specification",
        )
    output = Path(destination)
    if output.exists():
        raise ResultWriteError(
            f"Cross-fit sample-edge destination {output.name!r} already exists",
            code="crossfit_sample_edge_destination_exists",
            field="destination",
            remediation="Choose a new versioned artifact directory",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    _mark_incomplete(temporary)
    try:
        fold_records: list[dict[str, object]] = []
        ordered_folds = sorted(artifacts.folds, key=lambda item: item.fold_id)
        for index, fold in enumerate(ordered_folds):
            transform = fold.absolute_activity_v2_transform
            scores = fold.sample_edge_scores_v2
            if transform is None or scores is None:
                raise ValueError("configured cross-fit has an incomplete M0 v2 fold")
            directory = f"fold-{index:03d}"
            child = write_sample_edge_score_v2(temporary / directory, scores)
            fold_records.append(
                {
                    "directory": directory,
                    "fold_id": fold.fold_id,
                    "rows": len(scores.table),
                    "transform_manifest_id": transform.transform_manifest_id,
                    "provenance_id": scores.provenance.provenance_id,
                    "sample_edge_artifact_id": child.manifest["artifact_id"],
                }
            )
        payload: dict[str, object] = {
            "artifact_kind": CROSSFIT_SAMPLE_EDGE_V2_ARTIFACT_KIND,
            "artifact_schema_version": CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION,
            "status": "complete",
            "crossfit_id": artifacts.crossfit_id,
            "spec_id": artifacts.spec.spec_id,
            "repeat_id": artifacts.spec.repeat_id,
            "folds": fold_records,
        }
        artifact_id = stable_id(
            "crossfit_sample_edge_v2_artifact",
            payload,
            schema_version=CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION,
        )
        _write_json(
            temporary / _MANIFEST_FILENAME,
            {"artifact_id": artifact_id, **payload},
        )
        _write_json(
            temporary / _STATUS_FILENAME,
            {
                "artifact_id": artifact_id,
                "artifact_kind": CROSSFIT_SAMPLE_EDGE_V2_ARTIFACT_KIND,
                "artifact_schema_version": CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION,
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
            "Cross-fit sample-edge write failed and was marked incomplete",
            code="crossfit_sample_edge_write_failed",
            field="destination",
            remediation="Inspect the cause and rerun to a new destination",
        ) from error
    return CrossFitSampleEdgeV2Result.load(output)


__all__ = [
    "CROSSFIT_SAMPLE_EDGE_V2_ARTIFACT_KIND",
    "CROSSFIT_SAMPLE_EDGE_V2_SCHEMA_VERSION",
    "CrossFitSampleEdgeV2Result",
    "write_crossfit_sample_edge_v2_result",
]
