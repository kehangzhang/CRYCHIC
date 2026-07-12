"""Validate one external-method benchmark bundle without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from benchmarks.adapters.common import (
    LONG_TABLE_COLUMNS,
    LONG_TABLE_SCHEMA,
    MANIFEST_SCHEMA,
    RESULT_STATUSES,
    sha256_file,
)

_IDENTIFIER_COLUMNS = (
    "run_id",
    "dataset_id",
    "method_id",
    "method_version",
    "analysis_track",
    "resource_mode",
    "resource_id",
    "resource_version",
    "universe_id",
    "universe_member",
    "universe_size",
    "sample_id",
    "subject_id",
    "context_json",
    "interaction_id",
    "score_name",
    "score_direction",
    "status",
)
_KEY_COLUMNS = ("sender", "receiver", "interaction_id", "target")
_READ_COLUMNS = (
    "schema_version",
    *_IDENTIFIER_COLUMNS,
    *_KEY_COLUMNS,
    "score",
    "differential_effect",
    "differential_p_value",
    "differential_q_value",
    "reason_code",
)


def _constant(table: pd.DataFrame, column: str) -> object:
    values = table[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{column} must be constant within one bundle")
    return values.iloc[0]


def _universe_digest(sample: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(
        sample.loc[:, list(_KEY_COLUMNS)], index=False
    ).sort_values()
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def _expected_run_rows(manifest: dict[str, Any]) -> dict[str, int]:
    source_result = manifest.get("source_result")
    if not isinstance(source_result, dict):
        return {str(manifest.get("run_id")): int(manifest["output"]["rows"])}
    score_views = source_result.get("score_views")
    if not isinstance(score_views, list) or not score_views:
        raise ValueError("source_result must declare non-empty score_views")
    expected: dict[str, int] = {}
    for view in score_views:
        if not isinstance(view, dict):
            raise ValueError("score_views entries must be objects")
        run_id = str(view.get("run_id", ""))
        if not run_id or run_id in expected:
            raise ValueError("score_views must declare unique non-empty run IDs")
        expected[run_id] = int(view.get("rows", -1))
    return expected


def validate_external_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate manifest integrity and the materialized fixed universe."""
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("external adapter manifest has an invalid schema version")
    if manifest.get("status") != "complete":
        raise ValueError("external adapter manifest is not complete")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("manifest is missing its output object")
    if output.get("schema_version") != LONG_TABLE_SCHEMA:
        raise ValueError("manifest output has an invalid schema version")
    table_path = bundle_dir / str(output.get("table", "interactions_long.parquet"))
    parquet = pq.ParquetFile(table_path)

    if tuple(parquet.schema_arrow.names) != LONG_TABLE_COLUMNS:
        raise ValueError("Parquet columns differ from LONG_TABLE_COLUMNS")
    if parquet.metadata.num_rows != int(output.get("rows", -1)):
        raise ValueError("manifest and Parquet row counts differ")
    observed_sha256 = sha256_file(table_path)
    if observed_sha256 != output.get("sha256"):
        raise ValueError("manifest and Parquet SHA256 values differ")

    table = pq.read_table(table_path, columns=list(dict.fromkeys(_READ_COLUMNS)))
    frame = table.to_pandas(types_mapper=pd.ArrowDtype)
    if frame.empty:
        raise ValueError("external benchmark bundle must not be empty")
    if frame[list(_IDENTIFIER_COLUMNS)].isna().any().any():
        raise ValueError("required identifiers contain null values")
    if str(_constant(frame, "schema_version")) != LONG_TABLE_SCHEMA:
        raise ValueError("external long table has an invalid schema version")
    expected_run_rows = _expected_run_rows(manifest)
    observed_run_rows = {
        str(run_id): int(count)
        for run_id, count in frame["run_id"].value_counts().items()
    }
    if observed_run_rows != expected_run_rows:
        raise ValueError("manifest score views and table run IDs/rows differ")
    if str(_constant(frame, "dataset_id")) != manifest.get("dataset_id"):
        raise ValueError("manifest and table dataset IDs differ")

    statuses = set(frame["status"].astype(str))
    invalid_statuses = statuses.difference(RESULT_STATUSES)
    if invalid_statuses:
        raise ValueError(f"invalid result statuses: {sorted(invalid_statuses)}")
    numeric_score = pd.to_numeric(frame["score"], errors="coerce")
    if not np.isfinite(numeric_score.dropna().to_numpy(dtype=float)).all():
        raise ValueError("native scores must be finite")
    ok = frame["status"].eq("ok")
    if numeric_score[ok].isna().any():
        raise ValueError("status='ok' requires a score")
    if numeric_score[~ok].notna().any():
        raise ValueError("non-ok statuses require a missing score")
    differential_columns = (
        "differential_effect",
        "differential_p_value",
        "differential_q_value",
    )
    if frame[list(differential_columns)].notna().any().any():
        raise ValueError("external adapters must not emit between-condition inference")
    if not frame["universe_member"].astype(bool).all():
        raise ValueError("all rows must belong to the frozen universe")

    duplicate_key = ["run_id", "sample_id", *_KEY_COLUMNS]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("external long table contains duplicate result keys")
    universe_size = int(str(_constant(frame, "universe_size")))
    rows_by_sample = frame.groupby(
        ["run_id", "sample_id"], observed=True, sort=True
    ).size()
    if not rows_by_sample.eq(universe_size).all():
        raise ValueError(
            "each run view and sample must materialize exactly universe_size rows"
        )
    key_hashes = frame.groupby(
        ["run_id", "sample_id"], observed=True, sort=True
    ).apply(
        _universe_digest,
        include_groups=False,
    )
    if key_hashes.nunique() != 1:
        raise ValueError("run views and samples do not share one frozen edge universe")

    status_counts = {
        str(status): int(count)
        for status, count in frame["status"].value_counts().items()
    }
    return {
        "status": "valid",
        "bundle": str(bundle_dir.resolve()),
        "dataset_id": str(_constant(frame, "dataset_id")),
        "method_id": str(_constant(frame, "method_id")),
        "analysis_track": str(_constant(frame, "analysis_track")),
        "resource_mode": str(_constant(frame, "resource_mode")),
        "rows": len(frame),
        "run_ids": sorted(expected_run_rows),
        "run_views": len(expected_run_rows),
        "samples": int(frame["sample_id"].nunique()),
        "subjects": int(frame["subject_id"].nunique()),
        "contexts": int(frame["context_json"].nunique()),
        "universe_id": str(_constant(frame, "universe_id")),
        "universe_size": universe_size,
        "status_counts": status_counts,
        "sample_failures": manifest.get("sample_failures", {}),
        "parquet_sha256": observed_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_external_bundle(args.bundle_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
