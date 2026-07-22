"""Export frozen LIANA component arms from one per-sample rank-aggregate run.

The source ``interactions_long.parquet`` is the authority for the fixed edge
universe and all structural statuses.  Raw LIANA rows supply scores only for
source rows whose status is ``ok``; absence is never converted to zero.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.adapters.common import (
    INFERENTIAL_REASON,
    LONG_TABLE_COLUMNS,
    LONG_TABLE_SCHEMA,
    RESULT_STATUSES,
    canonical_digest,
    git_metadata,
    json_safe,
    sha256_file,
)

MANIFEST_SCHEMA = "crychic-liana-component-export-manifest-v1"
RAW_KEYS = ("sample_id", "source", "target", "ligand_complex", "receptor_complex")
JOIN_KEYS = ("sample_id", "sender", "receiver", "ligand", "receptor")
SINGLETON_FIELDS = (
    "schema_version",
    "run_id",
    "dataset_id",
    "method_id",
    "method_version",
    "analysis_track",
    "resource_mode",
    "resource_id",
    "resource_version",
    "universe_id",
)
WITHIN_SAMPLE_P_VALUE_SEMANTICS = (
    "LIANA CellPhoneDB component cell-label specificity; "
    "not between-condition inference"
)


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One exported method arm backed by a LIANA raw score column."""

    method_id: str
    display_name: str
    raw_column: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("component direction must be 'higher' or 'lower'")


COMPONENT_SPECS = (
    ComponentSpec(
        "liana_cellphonedb_mean",
        "LIANA CellPhoneDB mean component",
        "lr_means",
        "higher",
    ),
    ComponentSpec(
        "liana_cellphonedb_pvalue",
        "LIANA CellPhoneDB p-value component",
        "cellphone_pvals",
        "lower",
    ),
    ComponentSpec(
        "liana_connectome",
        "LIANA Connectome specificity component",
        "scaled_weight",
        "higher",
    ),
    ComponentSpec(
        "liana_logfc",
        "LIANA logFC Mean component",
        "lr_logfc",
        "higher",
    ),
    ComponentSpec(
        "liana_natmi",
        "LIANA NATMI specificity component",
        "spec_weight",
        "higher",
    ),
    ComponentSpec(
        "liana_singlecellsignalr",
        "LIANA SingleCellSignalR component",
        "lrscore",
        "higher",
    ),
)
COMPONENT_COLUMNS = tuple(spec.raw_column for spec in COMPONENT_SPECS)
RAW_REQUIRED_COLUMNS = (*RAW_KEYS, *COMPONENT_COLUMNS)


@dataclass(frozen=True, slots=True)
class RawFile:
    path: Path
    sample_id: str
    rows: int
    bytes: int
    sha256: str


class RawSampleCache:
    """Small LRU cache so large cohorts do not retain every raw table in RAM."""

    def __init__(self, files: dict[str, RawFile], *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("raw cache capacity must be positive")
        self._files = files
        self._capacity = capacity
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, sample_id: str) -> pd.DataFrame | None:
        info = self._files.get(sample_id)
        if info is None:
            return None
        cached = self._cache.pop(sample_id, None)
        if cached is None:
            cached = _load_raw_sample(info)
        self._cache[sample_id] = cached
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return cached


def _strict_strings(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    if frame.loc[:, columns].isna().any().any():
        raise ValueError(f"{label} identifiers contain null values")
    for column in columns:
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise ValueError(f"{label} identifier {column!r} contains empty values")


def _index_raw_files(raw_dir: Path) -> dict[str, RawFile]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"LIANA raw directory does not exist: {raw_dir}")
    paths = sorted(raw_dir.glob("sample_*.parquet"))
    if not paths:
        paths = sorted(raw_dir.glob("*.parquet"))
    indexed: dict[str, RawFile] = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = set(RAW_REQUIRED_COLUMNS).difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path.name} is missing raw columns: {sorted(missing)}")
        rows = parquet.metadata.num_rows
        if rows < 1:
            raise ValueError(f"LIANA raw file is empty: {path}")
        sample_values = parquet.read(columns=["sample_id"])["sample_id"].to_pandas()
        if sample_values.isna().any():
            raise ValueError(f"{path.name} contains null sample IDs")
        samples = sample_values.astype(str).unique()
        if len(samples) != 1:
            raise ValueError(f"{path.name} must contain exactly one sample ID")
        sample_id = str(samples[0])
        if sample_id in indexed:
            raise ValueError(f"multiple raw files represent sample {sample_id!r}")
        indexed[sample_id] = RawFile(
            path=path,
            sample_id=sample_id,
            rows=rows,
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
    return indexed


def _load_raw_sample(info: RawFile) -> pd.DataFrame:
    raw = pd.read_parquet(info.path, columns=list(RAW_REQUIRED_COLUMNS))
    _strict_strings(raw, RAW_KEYS, label=f"raw sample {info.sample_id!r}")
    if set(raw["sample_id"]) != {info.sample_id}:
        raise ValueError(f"raw sample identity changed while reading {info.path.name}")
    if raw.duplicated(list(RAW_KEYS), keep=False).any():
        raise ValueError(f"raw sample {info.sample_id!r} contains duplicate edge keys")

    renamed = raw.rename(
        columns={
            "source": "sender",
            "target": "receiver",
            "ligand_complex": "ligand",
            "receptor_complex": "receptor",
        }
    )
    for spec in COMPONENT_SPECS:
        values = pd.to_numeric(renamed[spec.raw_column], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(
                f"raw sample {info.sample_id!r} has non-finite "
                f"{spec.raw_column!r} values"
            )
        renamed[spec.raw_column] = values.astype(float)
        renamed[f"__rank_{spec.raw_column}"] = values.rank(
            method="average", ascending=spec.direction == "lower"
        )
    columns = [
        *JOIN_KEYS,
        *COMPONENT_COLUMNS,
        *(f"__rank_{column}" for column in COMPONENT_COLUMNS),
    ]
    renamed["__raw_present"] = True
    return cast(
        pd.DataFrame,
        renamed.loc[:, [*columns, "__raw_present"]].copy(),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _publish_directory(staged: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    if not output.exists():
        os.replace(staged, output)
        return

    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink()


def _singleton_values(frame: pd.DataFrame, trackers: dict[str, set[str]]) -> None:
    for field in SINGLETON_FIELDS:
        if frame[field].isna().any():
            raise ValueError(f"template field {field!r} contains null values")
        trackers[field].update(frame[field].astype(str).unique())


def _validate_template_chunk(
    frame: pd.DataFrame,
    *,
    singleton_trackers: dict[str, set[str]],
    sample_rows: dict[str, int],
    sample_universe_sizes: dict[str, set[int]],
    status_counts: dict[str, int],
) -> None:
    _singleton_values(frame, singleton_trackers)
    if set(frame["schema_version"].astype(str)) != {LONG_TABLE_SCHEMA}:
        raise ValueError("template has an invalid interactions_long schema_version")
    statuses = frame["status"].astype(str)
    invalid = set(statuses).difference(RESULT_STATUSES)
    if invalid:
        raise ValueError(f"template contains invalid statuses: {sorted(invalid)}")
    for status, count in statuses.value_counts().items():
        status_counts[str(status)] = status_counts.get(str(status), 0) + int(count)

    identifiers = [
        "sample_id",
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "status",
        "reason_code",
    ]
    if frame.loc[:, identifiers].isna().any().any():
        raise ValueError("template contains null required edge/status identifiers")
    numeric_score = pd.to_numeric(frame["score"], errors="coerce")
    supplied = frame["score"].notna()
    if (supplied & numeric_score.isna()).any() or np.isinf(
        numeric_score.fillna(0).to_numpy(dtype=float)
    ).any():
        raise ValueError("template contains a non-finite source score")
    ok = statuses.eq("ok")
    if numeric_score.loc[ok].isna().any() or numeric_score.loc[~ok].notna().any():
        raise ValueError("template score/status semantics are inconsistent")
    for field in (
        "differential_effect",
        "differential_p_value",
        "differential_q_value",
    ):
        if frame[field].notna().any():
            raise ValueError(f"template {field} must be null for a per-sample run")

    universe_sizes = pd.to_numeric(frame["universe_size"], errors="coerce")
    if universe_sizes.isna().any() or (universe_sizes < 1).any():
        raise ValueError("template universe_size must be positive")
    if (universe_sizes % 1 != 0).any():
        raise ValueError("template universe_size must be integral")
    for sample_id, indexes in frame.groupby("sample_id", sort=False).groups.items():
        sample = str(sample_id)
        sample_rows[sample] = sample_rows.get(sample, 0) + len(indexes)
        sample_universe_sizes.setdefault(sample, set()).update(
            universe_sizes.loc[indexes].astype(int).unique()
        )


def _component_run_ids(
    template_sha256: str, raw_files: dict[str, RawFile]
) -> dict[str, str]:
    raw_digests = [
        {"sample_id": item.sample_id, "sha256": item.sha256}
        for item in sorted(raw_files.values(), key=lambda value: value.sample_id)
    ]
    return {
        spec.method_id: canonical_digest(
            {
                "schema_version": MANIFEST_SCHEMA,
                "template_sha256": template_sha256,
                "raw_files": raw_digests,
                "method_id": spec.method_id,
                "score_column": spec.raw_column,
                "score_direction": spec.direction,
            },
            prefix="external_component_run",
        )
        for spec in COMPONENT_SPECS
    }


def _join_raw(
    template: pd.DataFrame,
    raw: pd.DataFrame | None,
    *,
    sample_id: str,
) -> tuple[pd.DataFrame, int]:
    if raw is None:
        joined = template.copy()
        for column in COMPONENT_COLUMNS:
            joined[column] = np.nan
            joined[f"__rank_{column}"] = np.nan
        joined["__raw_present"] = False
    else:
        joined = template.merge(
            raw,
            on=list(JOIN_KEYS),
            how="left",
            sort=False,
            validate="many_to_one",
        )
        joined["__raw_present"] = joined["__raw_present"].eq(True)

    ok = joined["status"].astype(str).eq("ok")
    present = joined["__raw_present"].astype(bool)
    if (ok & ~present).any():
        raise ValueError(
            f"template status='ok' row is absent from raw sample {sample_id!r}"
        )
    if (~ok & present).any():
        raise ValueError(
            f"raw sample {sample_id!r} maps to a non-ok structural template row"
        )
    return joined, int(present.sum())


def _component_table(
    joined: pd.DataFrame,
    spec: ComponentSpec,
    *,
    run_id: str,
) -> pd.DataFrame:
    result = cast(pd.DataFrame, joined.loc[:, LONG_TABLE_COLUMNS].copy())
    ok = result["status"].astype(str).eq("ok")
    result["run_id"] = run_id
    result["method_id"] = spec.method_id
    result["score"] = pd.to_numeric(joined[spec.raw_column], errors="coerce")
    result.loc[~ok, "score"] = np.nan
    result["score_name"] = spec.raw_column
    result["score_direction"] = spec.direction
    result["rank"] = pd.to_numeric(joined[f"__rank_{spec.raw_column}"], errors="coerce")
    result.loc[~ok, "rank"] = np.nan
    result["specificity_score"] = np.nan
    result["specificity_score_name"] = pd.NA
    if spec.raw_column == "cellphone_pvals":
        result["within_dataset_p_value"] = result["score"]
        result["within_dataset_p_value_semantics"] = WITHIN_SAMPLE_P_VALUE_SEMANTICS
    else:
        result["within_dataset_p_value"] = np.nan
        result["within_dataset_p_value_semantics"] = "not_emitted"
    result["differential_effect"] = np.nan
    result["differential_p_value"] = np.nan
    result["differential_q_value"] = np.nan
    return result


def _raw_manifest_records(raw_files: dict[str, RawFile]) -> list[dict[str, Any]]:
    return [
        {
            "filename": item.path.name,
            "sample_id": item.sample_id,
            "rows": item.rows,
            "bytes": item.bytes,
            "sha256": item.sha256,
        }
        for item in sorted(raw_files.values(), key=lambda value: value.sample_id)
    ]


def run(
    raw_dir: Path,
    template_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    batch_size: int = 100_000,
    raw_cache_size: int = 2,
) -> dict[str, Any]:
    """Export six fixed-universe component bundles and return the root manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    raw_dir = raw_dir.resolve()
    template_path = template_path.resolve()
    output_dir = output_dir.resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"template parquet does not exist: {template_path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_dir}; pass --overwrite to replace it"
        )

    started = time.perf_counter()
    raw_files = _index_raw_files(raw_dir)
    template_sha256 = sha256_file(template_path)
    run_ids = _component_run_ids(template_sha256, raw_files)
    source_manifest = raw_dir.parent / "manifest.json"
    source_manifest_record = (
        {
            "filename": source_manifest.name,
            "sha256": sha256_file(source_manifest),
            "bytes": source_manifest.stat().st_size,
        }
        if source_manifest.is_file()
        else None
    )

    parquet = pq.ParquetFile(template_path)
    template_columns = tuple(parquet.schema_arrow.names)
    missing = set(LONG_TABLE_COLUMNS).difference(template_columns)
    extra = set(template_columns).difference(LONG_TABLE_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"template columns differ from canonical schema: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    template_schema = parquet.schema_arrow

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    writers: dict[str, pq.ParquetWriter] = {}
    published = False
    try:
        for spec in COMPONENT_SPECS:
            method_dir = staged / spec.method_id
            method_dir.mkdir()
            writers[spec.method_id] = pq.ParquetWriter(
                method_dir / "interactions_long.parquet",
                template_schema,
                compression="zstd",
                use_dictionary=True,
            )

        singleton_trackers = {field: set() for field in SINGLETON_FIELDS}
        sample_rows: dict[str, int] = {}
        sample_universe_sizes: dict[str, set[int]] = {}
        status_counts: dict[str, int] = {}
        raw_match_counts = {sample_id: 0 for sample_id in raw_files}
        cache = RawSampleCache(raw_files, capacity=raw_cache_size)
        total_rows = 0

        for batch in parquet.iter_batches(batch_size=batch_size):
            chunk = batch.to_pandas()
            _validate_template_chunk(
                chunk,
                singleton_trackers=singleton_trackers,
                sample_rows=sample_rows,
                sample_universe_sizes=sample_universe_sizes,
                status_counts=status_counts,
            )
            total_rows += len(chunk)
            for sample_value, indexes in chunk.groupby(
                "sample_id", sort=False, observed=True
            ).groups.items():
                sample_id = str(sample_value)
                template_sample = cast(pd.DataFrame, chunk.loc[indexes].copy())
                joined, matched = _join_raw(
                    template_sample,
                    cache.get(sample_id),
                    sample_id=sample_id,
                )
                if sample_id in raw_match_counts:
                    raw_match_counts[sample_id] += matched
                for spec in COMPONENT_SPECS:
                    output = _component_table(
                        joined, spec, run_id=run_ids[spec.method_id]
                    )
                    arrow = pa.Table.from_pandas(
                        output,
                        schema=template_schema,
                        preserve_index=False,
                        safe=True,
                    )
                    writers[spec.method_id].write_table(
                        arrow, row_group_size=min(batch_size, len(output))
                    )

        for writer in writers.values():
            writer.close()
        writers.clear()

        if total_rows != parquet.metadata.num_rows:
            raise RuntimeError(
                "template stream row count differs from parquet metadata"
            )
        for field, values in singleton_trackers.items():
            if len(values) != 1:
                raise ValueError(
                    f"template field {field!r} must have one value, "
                    f"found {sorted(values)}"
                )
        if singleton_trackers["schema_version"] != {LONG_TABLE_SCHEMA}:
            raise ValueError("template schema_version is not canonical")
        for sample_id, count in sample_rows.items():
            sizes = sample_universe_sizes[sample_id]
            if len(sizes) != 1 or count != next(iter(sizes)):
                raise ValueError(
                    f"template sample {sample_id!r} does not materialize universe_size"
                )
        unknown_raw_samples = set(raw_files).difference(sample_rows)
        if unknown_raw_samples:
            raise ValueError(
                "raw samples are absent from the template: "
                f"{sorted(unknown_raw_samples)}"
            )
        for sample_id, info in raw_files.items():
            if raw_match_counts[sample_id] != info.rows:
                raise ValueError(
                    f"raw sample {sample_id!r} matched {raw_match_counts[sample_id]} "
                    f"of {info.rows} rows in the template"
                )

        source_values = {
            field: next(iter(values)) for field, values in singleton_trackers.items()
        }
        repo_root = Path(__file__).resolve().parents[2]
        raw_records = _raw_manifest_records(raw_files)
        component_records: dict[str, dict[str, Any]] = {}
        for spec in COMPONENT_SPECS:
            method_dir = staged / spec.method_id
            table_path = method_dir / "interactions_long.parquet"
            table_record = {
                "filename": table_path.name,
                "rows": total_rows,
                "bytes": table_path.stat().st_size,
                "sha256": sha256_file(table_path),
                "schema_version": LONG_TABLE_SCHEMA,
            }
            component_manifest: dict[str, Any] = {
                "schema_version": MANIFEST_SCHEMA,
                "kind": "component_bundle",
                "status": "complete",
                "run_id": run_ids[spec.method_id],
                "source_run_id": source_values["run_id"],
                "dataset_id": source_values["dataset_id"],
                "method": {
                    "id": spec.method_id,
                    "name": spec.display_name,
                    "version": source_values["method_version"],
                    "derived_from_method_id": source_values["method_id"],
                },
                "score_semantics": {
                    "raw_column": spec.raw_column,
                    "direction": spec.direction,
                    "rank": "average ties within biological sample",
                    "structural_statuses": "copied exactly from frozen template",
                    "non_ok_scores": None,
                },
                "statistical_scope": {
                    "unit": "biological sample",
                    "within_dataset_p_value": (
                        WITHIN_SAMPLE_P_VALUE_SEMANTICS
                        if spec.raw_column == "cellphone_pvals"
                        else None
                    ),
                    "differential_effect": None,
                    "differential_p_value": None,
                    "differential_q_value": None,
                    "reason_code_for_observed_rows": INFERENTIAL_REASON,
                },
                "source": {
                    "template": {
                        "filename": template_path.name,
                        "bytes": template_path.stat().st_size,
                        "sha256": template_sha256,
                    },
                    "raw_files": raw_records,
                    "source_manifest": source_manifest_record,
                },
                "status_counts": status_counts,
                "output": table_record,
                "code": {
                    **git_metadata(repo_root),
                    "entrypoint": str(Path(__file__).relative_to(repo_root)),
                    "entrypoint_sha256": sha256_file(__file__),
                },
            }
            manifest_path = method_dir / "manifest.json"
            _atomic_json(manifest_path, component_manifest)
            component_records[spec.method_id] = {
                "method_id": spec.method_id,
                "score_column": spec.raw_column,
                "score_direction": spec.direction,
                "directory": spec.method_id,
                "run_id": run_ids[spec.method_id],
                "table": table_record,
                "manifest": {
                    "filename": manifest_path.name,
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                },
            }

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "kind": "component_export",
            "status": "complete",
            "source_run_id": source_values["run_id"],
            "dataset_id": source_values["dataset_id"],
            "elapsed_seconds": time.perf_counter() - started,
            "template": {
                "filename": template_path.name,
                "rows": total_rows,
                "bytes": template_path.stat().st_size,
                "sha256": template_sha256,
                "status_counts": status_counts,
            },
            "raw_files": raw_records,
            "source_manifest": source_manifest_record,
            "components": component_records,
            "statistical_scope": {
                "unit": "biological sample",
                "differential_effect": None,
                "differential_p_value": None,
                "differential_q_value": None,
                "cellphone_pvals_note": WITHIN_SAMPLE_P_VALUE_SEMANTICS,
            },
            "code": {
                **git_metadata(repo_root),
                "entrypoint": str(Path(__file__).relative_to(repo_root)),
                "entrypoint_sha256": sha256_file(__file__),
            },
        }
        _atomic_json(staged / "manifest.json", manifest)
        _publish_directory(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        for writer in writers.values():
            writer.close()
        if not published and staged.exists():
            shutil.rmtree(staged)


def _resolve_path_argument(
    parser: argparse.ArgumentParser,
    positional: Path | None,
    optional: Path | None,
    *,
    label: str,
) -> Path:
    if positional is not None and optional is not None:
        parser.error(f"provide {label} either positionally or by flag, not both")
    value = optional if optional is not None else positional
    if value is None:
        parser.error(f"missing required {label}")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir_pos", nargs="?", type=Path)
    parser.add_argument("template_pos", nargs="?", type=Path)
    parser.add_argument("output_dir_pos", nargs="?", type=Path)
    parser.add_argument("--raw-dir", dest="raw_dir", type=Path)
    parser.add_argument(
        "--template",
        "--interactions-long",
        dest="template_path",
        type=Path,
    )
    parser.add_argument("--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--raw-cache-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    raw_dir = _resolve_path_argument(
        parser, args.raw_dir_pos, args.raw_dir, label="raw directory"
    )
    template_path = _resolve_path_argument(
        parser, args.template_pos, args.template_path, label="template parquet"
    )
    output_dir = _resolve_path_argument(
        parser, args.output_dir_pos, args.output_dir, label="output directory"
    )
    manifest = run(
        raw_dir,
        template_path,
        output_dir,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        raw_cache_size=args.raw_cache_size,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_id": manifest["dataset_id"],
                "components": len(cast(dict[str, Any], manifest["components"])),
                "output_dir": str(output_dir.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
