"""Shared contracts and provenance helpers for external benchmark adapters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

LONG_TABLE_SCHEMA = "crychic-external-interactions-long-v1"
MANIFEST_SCHEMA = "crychic-external-adapter-manifest-v1"
INFERENTIAL_REASON = "external_adapter_no_between_condition_inference"
RESOURCE_MODES = frozenset({"H-common", "H-covered", "native"})
RESULT_STATUSES = frozenset(
    {
        "ok",
        "not_returned",
        "resource_unavailable",
        "unsupported_resource",
        "insufficient_cells",
        "method_failed",
        "missing",
    }
)

LONG_TABLE_COLUMNS = (
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
    "universe_member",
    "universe_size",
    "sample_id",
    "subject_id",
    "context_json",
    "sender",
    "receiver",
    "interaction_id",
    "native_interaction_id",
    "interaction_direction",
    "ligand",
    "receptor",
    "target",
    "score",
    "score_name",
    "score_direction",
    "rank",
    "specificity_score",
    "specificity_score_name",
    "within_dataset_p_value",
    "within_dataset_p_value_semantics",
    "differential_effect",
    "differential_p_value",
    "differential_q_value",
    "status",
    "reason_code",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any], *, prefix: str = "run") -> str:
    """Return a stable identifier derived from canonical JSON."""
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:32]}"


def json_safe(value: Any) -> Any:
    """Convert common numerical values into strict-JSON-compatible objects."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write strict, sorted JSON."""
    Path(path).write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Record Git identity without failing outside a worktree."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Read installed distribution versions."""
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def python_environment(
    *, environment_name: str, packages: Iterable[str], threads: int
) -> dict[str, Any]:
    """Describe the active Python process and requested thread budget."""
    return {
        "environment_name": environment_name,
        "runtime": "python",
        "executable_name": Path(sys.executable).name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "threads": int(threads),
        "packages": package_versions(packages),
    }


def canonical_context_json(row: Mapping[str, Any], context_keys: Sequence[str]) -> str:
    """Serialize one sample context with sorted keys."""
    return json.dumps(
        {key: str(row[key]) for key in context_keys},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_prepared_input(
    adata: Any,
    *,
    sample_key: str,
    subject_key: str,
    cell_type_key: str,
    context_keys: Sequence[str],
) -> pd.DataFrame:
    """Validate the metadata needed by every external adapter."""
    required = {sample_key, subject_key, cell_type_key, *context_keys}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"prepared h5ad is missing metadata: {sorted(missing)}")
    if not adata.obs_names.is_unique:
        raise ValueError("prepared h5ad observation identifiers must be unique")
    for key in required:
        if adata.obs[key].isna().any():
            raise ValueError(f"prepared h5ad metadata {key!r} contains null values")
        if adata.obs[key].astype(str).str.contains("\\|", regex=True).any():
            raise ValueError(
                f"prepared h5ad metadata {key!r} contains reserved '|' values"
            )
    columns = [sample_key, subject_key, *context_keys]
    samples = adata.obs[columns].astype(str).drop_duplicates()
    duplicated = samples[sample_key].duplicated(keep=False)
    if duplicated.any():
        affected = sorted(samples.loc[duplicated, sample_key].unique())
        raise ValueError(
            f"sample IDs do not map to one subject/context tuple: {affected[:5]}"
        )
    samples = samples.rename(
        columns={sample_key: "sample_id", subject_key: "subject_id"}
    ).sort_values("sample_id", kind="stable", ignore_index=True)
    samples["context_json"] = [
        canonical_context_json(row, context_keys)
        for row in samples.to_dict(orient="records")
    ]
    return cast(
        pd.DataFrame,
        samples.loc[:, ["sample_id", "subject_id", "context_json"]].copy(),
    )


def cell_type_support(
    adata: Any, *, sample_key: str, cell_type_key: str
) -> pd.DataFrame:
    """Count cells for every observed sample/cell-type combination."""
    frame = pd.DataFrame(
        {
            "sample_id": adata.obs[sample_key].astype(str).to_numpy(),
            "cell_type": adata.obs[cell_type_key].astype(str).to_numpy(),
        }
    )
    return (
        frame.groupby(["sample_id", "cell_type"], sort=True, observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )


def stable_interaction_id(
    *, resource_id: str, ligand: str, receptor: str, native_id: str
) -> str:
    """Create a stable method-resource interaction ID."""
    return canonical_digest(
        {
            "resource_id": resource_id,
            "ligand": ligand,
            "receptor": receptor,
            "native_id": native_id,
        },
        prefix="external_interaction",
    )


def frozen_universe_id(
    *,
    dataset_id: str,
    resource_mode: str,
    interactions: pd.DataFrame,
    cell_types: Sequence[str],
    sender_types: Sequence[str] | None = None,
    receiver_types: Sequence[str] | None = None,
) -> str:
    """Hash the exact edge and cell-type ontology used by every sample."""
    if resource_mode not in RESOURCE_MODES:
        raise ValueError(f"unsupported resource_mode: {resource_mode!r}")
    required = {"interaction_id", "ligand", "receptor"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"frozen resource is missing columns: {sorted(missing)}")
    edges = (
        interactions.loc[:, sorted(required)]
        .astype(str)
        .sort_values(sorted(required), kind="stable", ignore_index=True)
    )
    payload: dict[str, Any] = {
        "schema_version": LONG_TABLE_SCHEMA,
        "dataset_id": dataset_id,
        "resource_mode": resource_mode,
        "cell_types": sorted(map(str, cell_types)),
        "edges": edges.to_dict(orient="records"),
    }
    if (sender_types is None) != (receiver_types is None):
        raise ValueError("sender_types and receiver_types must be declared together")
    if sender_types is not None and receiver_types is not None:
        payload["sender_types"] = sorted(map(str, sender_types))
        payload["receiver_types"] = sorted(map(str, receiver_types))
    return canonical_digest(payload, prefix="external_universe")


def normalize_frozen_resource(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the method-independent resource columns used for materialization."""
    required = {"interaction_id", "ligand", "receptor", "native_interaction_id"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"frozen resource is missing columns: {sorted(missing)}")
    result = table.copy()
    if "method_covered" not in result:
        result["method_covered"] = True
    if "input_available" not in result:
        result["input_available"] = True
    if "target" not in result:
        result["target"] = pd.NA
    for field in ("interaction_id", "ligand", "receptor"):
        if result[field].isna().any():
            raise ValueError(f"frozen resource field {field!r} contains nulls")
        result[field] = result[field].astype(str)
    result["native_interaction_id"] = result["native_interaction_id"].astype("string")
    result["method_covered"] = result["method_covered"].astype(bool)
    result["input_available"] = result["input_available"].astype(bool)
    duplicate_key = ["interaction_id", "target"]
    if result.duplicated(duplicate_key).any():
        raise ValueError("frozen resource contains duplicate interaction/target rows")
    return result.sort_values(
        ["ligand", "receptor", "interaction_id", "target"],
        kind="stable",
        ignore_index=True,
    )


def _empty_observation_columns() -> tuple[str, ...]:
    return (
        "score",
        "specificity_score",
        "within_dataset_p_value",
        "within_dataset_p_value_semantics",
    )


def materialize_fixed_universe(
    observed: pd.DataFrame,
    *,
    sample_metadata: pd.DataFrame,
    support: pd.DataFrame,
    resource: pd.DataFrame,
    dataset_id: str,
    run_id: str,
    method_id: str,
    method_version: str,
    analysis_track: str,
    resource_mode: str,
    resource_id: str,
    resource_version: str,
    score_name: str,
    score_direction: str,
    specificity_score_name: str | None,
    min_cells: int,
    sample_status: Mapping[str, str] | None = None,
    sender_types: Sequence[str] | None = None,
    sender_requires_cells: bool = True,
    interaction_direction: str = "ligand_to_receptor",
    validate: bool = True,
) -> pd.DataFrame:
    """Outer-join sparse native results to the exact sample-level universe.

    ``observed`` may contain only rows emitted by a method. This function makes
    absence explicit and never converts it to a native score of zero.
    """
    if resource_mode not in RESOURCE_MODES:
        raise ValueError(f"unsupported resource_mode: {resource_mode!r}")
    if score_direction not in {"higher", "lower"}:
        raise ValueError("score_direction must be 'higher' or 'lower'")
    if min_cells < 1:
        raise ValueError("min_cells must be positive")
    frozen = normalize_frozen_resource(resource)
    cell_types = tuple(sorted(support["cell_type"].astype(str).unique()))
    if not cell_types:
        raise ValueError("the prepared input contains no cell types")
    senders = (
        cell_types
        if sender_types is None
        else tuple(sorted(set(map(str, sender_types))))
    )
    if not senders:
        raise ValueError("sender_types must contain at least one value")
    pair_table = pd.DataFrame(
        product(senders, cell_types), columns=["sender", "receiver"]
    )
    pair_table["_join"] = 1
    frozen["_join"] = 1
    base = pair_table.merge(frozen, on="_join", how="inner").drop(columns="_join")
    universe_size = len(base)
    universe_id = frozen_universe_id(
        dataset_id=dataset_id,
        resource_mode=resource_mode,
        interactions=frozen,
        cell_types=cell_types,
        sender_types=None if sender_types is None else senders,
        receiver_types=None if sender_types is None else cell_types,
    )

    observation_key = ["sample_id", "sender", "receiver", "interaction_id", "target"]
    native = observed.copy()
    if "target" not in native:
        native["target"] = pd.NA
    missing_observed = set(observation_key).difference(native.columns)
    if missing_observed:
        raise ValueError(
            f"observed result is missing columns: {sorted(missing_observed)}"
        )
    for field in _empty_observation_columns():
        if field not in native:
            native[field] = pd.NA
    if native.duplicated(observation_key).any():
        raise ValueError("observed result contains duplicate frozen-universe keys")
    native = native.loc[:, [*observation_key, *_empty_observation_columns()]]

    counts = {
        (str(row.sample_id), str(row.cell_type)): int(float(str(row.n_cells)))
        for row in support.itertuples(index=False)
    }
    status_override = {} if sample_status is None else dict(sample_status)
    invalid = set(status_override.values()).difference(
        {"method_failed", "unsupported_resource"}
    )
    if invalid:
        raise ValueError(f"invalid sample status overrides: {sorted(invalid)}")

    frames: list[pd.DataFrame] = []
    for sample in sample_metadata.itertuples(index=False):
        sample_id = str(sample.sample_id)
        frame = base.copy()
        frame.insert(0, "sample_id", sample_id)
        frame = frame.merge(
            native, on=observation_key, how="left", validate="one_to_one"
        )
        enough_cells = np.fromiter(
            (
                (
                    not sender_requires_cells
                    or counts.get((sample_id, sender), 0) >= min_cells
                )
                and counts.get((sample_id, receiver), 0) >= min_cells
                for sender, receiver in frame[["sender", "receiver"]].itertuples(
                    index=False, name=None
                )
            ),
            dtype=bool,
            count=len(frame),
        )
        covered = frame["method_covered"].to_numpy(dtype=bool)
        input_available = frame["input_available"].to_numpy(dtype=bool)
        has_score = pd.to_numeric(frame["score"], errors="coerce").notna().to_numpy()
        frame["status"] = np.select(
            [~covered, ~input_available, ~enough_cells, has_score],
            ["resource_unavailable", "missing", "insufficient_cells", "ok"],
            default="not_returned",
        )
        override = status_override.get(sample_id)
        if override is not None:
            eligible = covered & enough_cells
            frame.loc[eligible, "status"] = override
        not_ok = ~frame["status"].eq("ok")
        frame.loc[
            not_ok,
            ["score", "specificity_score", "within_dataset_p_value"],
        ] = np.nan
        frame["subject_id"] = str(sample.subject_id)
        frame["context_json"] = str(sample.context_json)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)

    table["schema_version"] = LONG_TABLE_SCHEMA
    table["run_id"] = run_id
    table["dataset_id"] = dataset_id
    table["method_id"] = method_id
    table["method_version"] = method_version
    table["analysis_track"] = analysis_track
    table["resource_mode"] = resource_mode
    table["resource_id"] = resource_id
    table["resource_version"] = resource_version
    table["universe_id"] = universe_id
    table["universe_member"] = True
    table["universe_size"] = universe_size
    table["interaction_direction"] = interaction_direction
    table["score_name"] = score_name
    table["score_direction"] = score_direction
    table["specificity_score_name"] = specificity_score_name
    table["differential_effect"] = np.nan
    table["differential_p_value"] = np.nan
    table["differential_q_value"] = np.nan
    table["reason_code"] = np.where(
        table["status"].eq("ok"), INFERENTIAL_REASON, table["status"]
    )
    observed_mask = table["status"].eq("ok")
    table["rank"] = np.nan
    for _, index in table.loc[observed_mask].groupby("sample_id").groups.items():
        table.loc[index, "rank"] = table.loc[index, "score"].rank(
            ascending=score_direction == "lower", method="average"
        )
    ordered = cast(pd.DataFrame, table.loc[:, LONG_TABLE_COLUMNS].copy())
    return validate_long_table(ordered) if validate else ordered


def empty_long_table() -> pd.DataFrame:
    """Return an empty table with the canonical column order."""
    return pd.DataFrame(columns=LONG_TABLE_COLUMNS)


def validate_long_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and order one adapter's normalized interaction table."""
    missing = set(LONG_TABLE_COLUMNS).difference(table.columns)
    extra = set(table.columns).difference(LONG_TABLE_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"external long table columns differ: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    table = cast(pd.DataFrame, table.loc[:, LONG_TABLE_COLUMNS].copy())
    if not table.empty:
        if set(table["schema_version"].astype(str)) != {LONG_TABLE_SCHEMA}:
            raise ValueError("external long table has an invalid schema_version")
        required = [
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
        ]
        if table[required].isna().any().any():
            raise ValueError("external long table has null required identifiers")
        modes = set(table["resource_mode"].astype(str))
        if not modes.issubset(RESOURCE_MODES):
            raise ValueError(f"external long table has invalid resource modes: {modes}")
        statuses = set(table["status"].astype(str))
        invalid_statuses = statuses.difference(RESULT_STATUSES)
        if invalid_statuses:
            raise ValueError(
                f"external long table has invalid statuses: {sorted(invalid_statuses)}"
            )
        if not set(table["score_direction"].astype(str)).issubset({"higher", "lower"}):
            raise ValueError("score_direction must be 'higher' or 'lower'")
        if not table["universe_member"].astype(bool).all():
            raise ValueError("all exported rows must belong to the frozen universe")
        universe_size = pd.to_numeric(table["universe_size"], errors="coerce")
        if universe_size.isna().any() or (universe_size < 1).any():
            raise ValueError("universe_size must be a positive integer")
        if (universe_size % 1 != 0).any():
            raise ValueError("universe_size must be integral")
        size_by_universe = table.groupby("universe_id", observed=True)[
            "universe_size"
        ].nunique()
        if (size_by_universe != 1).any():
            raise ValueError("universe_size must be constant within universe_id")
        for value in table["context_json"].astype(str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("context_json must encode an object")
        for field in (
            "differential_effect",
            "differential_p_value",
            "differential_q_value",
        ):
            if table[field].notna().any():
                raise ValueError(
                    f"{field} must remain null: adapters do not perform group inference"
                )
        numeric_score = pd.to_numeric(table["score"], errors="coerce")
        supplied = table["score"].notna()
        if (supplied & numeric_score.isna()).any() or np.isinf(
            numeric_score.fillna(0.0)
        ).any():
            raise ValueError("native scores must be finite numeric values")
        ok = table["status"].eq("ok")
        if numeric_score[ok].isna().any():
            raise ValueError("status='ok' requires a native score")
        if numeric_score[~ok].notna().any():
            raise ValueError("non-ok statuses require a missing native score")
        if table.duplicated(
            [
                "run_id",
                "sample_id",
                "sender",
                "receiver",
                "interaction_id",
                "target",
            ],
            keep=False,
        ).any():
            raise ValueError("external long table contains duplicate result keys")
        universe_keys = ["sender", "receiver", "interaction_id", "target"]
        for _, run in table.groupby("run_id", sort=False, observed=True):
            expected_keys: frozenset[tuple[object, ...]] | None = None
            for _, sample in run.groupby("sample_id", sort=False, observed=True):
                expected_size = int(sample["universe_size"].iloc[0])
                if len(sample) != expected_size:
                    raise ValueError(
                        "each sample must materialize exactly universe_size rows"
                    )
                observed_keys = frozenset(
                    sample.loc[:, universe_keys].itertuples(index=False, name=None)
                )
                if expected_keys is None:
                    expected_keys = observed_keys
                elif observed_keys != expected_keys:
                    raise ValueError(
                        "every sample must materialize the same frozen edge universe"
                    )
    return table


def load_harmonized_resource(
    table_path: str | Path, manifest_path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and checksum-verify the generated harmonized LR table."""
    table_path = Path(table_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payloads = [manifest.get("payload", {}), manifest.get("covered_payload", {})]
    matched = [
        payload for payload in payloads if payload.get("filename") == table_path.name
    ]
    if len(matched) != 1:
        raise ValueError(
            f"harmonized resource filename is not pinned by manifest: {table_path.name}"
        )
    expected = matched[0].get("sha256")
    observed = sha256_file(table_path)
    if expected != observed:
        raise ValueError(
            "harmonized resource checksum mismatch: "
            f"expected={expected}, observed={observed}"
        )
    table = pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"harmonized_interaction_id", "ligand", "receptor"}
    missing = required.difference(table.columns)
    if missing or table.empty:
        raise ValueError(f"harmonized resource is invalid: missing={sorted(missing)}")
    source_columns = [
        column
        for column in table.columns
        if column.endswith("_source_interaction_id")
    ]
    if not source_columns:
        raise ValueError("harmonized resource has no source interaction columns")
    if table.loc[:, source_columns].astype(str).eq("").all(axis=1).any():
        raise ValueError("harmonized resource contains a row with no source evidence")
    if table.duplicated(["ligand", "receptor"]).any():
        raise ValueError("harmonized resource contains duplicate ligand-receptor pairs")
    return table, manifest


def method_frozen_resource(
    table: pd.DataFrame,
    *,
    method: str,
    resource_mode: str,
) -> pd.DataFrame:
    """Map a harmonized common/covered table to one method's frozen resource."""
    if resource_mode not in {"H-common", "H-covered"}:
        raise ValueError("method_frozen_resource only accepts harmonized arms")
    source_column = f"{method}_source_interaction_id"
    if source_column not in table:
        raise ValueError(f"harmonized resource lacks {source_column!r}")
    coverage_column = f"{method}_covered"
    covered = (
        table[coverage_column].astype(str).str.lower().eq("true")
        if coverage_column in table
        else table[source_column].astype(str).ne("")
    )
    if resource_mode == "H-common" and not covered.all():
        raise ValueError("H-common contains an edge unavailable to the selected method")
    return pd.DataFrame(
        {
            "interaction_id": table["harmonized_interaction_id"].astype(str),
            "native_interaction_id": table[source_column].astype("string"),
            "ligand": table["ligand"].astype(str),
            "receptor": table["receptor"].astype(str),
            "method_covered": covered.to_numpy(dtype=bool),
        }
    )


def prepare_output(output_dir: str | Path, *, overwrite: bool) -> Path:
    """Create a clean adapter output directory."""
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {output}; pass --overwrite to replace it"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def begin_manifest(
    *,
    repo_root: str | Path,
    dataset_id: str,
    method: Mapping[str, Any],
    environment: Mapping[str, Any],
    input_path: str | Path,
    input_shape: Sequence[int],
    sample_metadata: pd.DataFrame,
    input_keys: Mapping[str, Any],
    resource: Mapping[str, Any],
    parameters: Mapping[str, Any],
    score_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a running manifest and stable run ID."""
    input_path = Path(input_path)
    run_spec = {
        "dataset_id": dataset_id,
        "method": dict(method),
        "input_sha256": sha256_file(input_path),
        "resource": dict(resource),
        "parameters": dict(parameters),
    }
    run_id = canonical_digest(run_spec, prefix="external_run")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "status": "running",
        "method": dict(method),
        "environment": dict(environment),
        "input": {
            "filename": input_path.name,
            "sha256": run_spec["input_sha256"],
            "bytes": input_path.stat().st_size,
            "shape": [int(value) for value in input_shape],
            "samples": int(sample_metadata["sample_id"].nunique()),
            "subjects": int(sample_metadata["subject_id"].nunique()),
            "contexts": int(sample_metadata["context_json"].nunique()),
            "keys": dict(input_keys),
        },
        "resource": dict(resource),
        "parameters": dict(parameters),
        "score_semantics": dict(score_semantics),
        "statistical_scope": {
            "unit": "biological sample",
            "within_dataset_p_value": score_semantics.get(
                "within_dataset_p_value", "not_emitted"
            ),
            "differential_effect": None,
            "differential_p_value": None,
            "differential_q_value": None,
            "reason_code": INFERENTIAL_REASON,
        },
        "code": git_metadata(repo_root),
        "started_unix_seconds": time.time(),
        "output": None,
        "failure": None,
    }


def finalize_manifest(
    manifest: dict[str, Any],
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    started: float,
) -> dict[str, Any]:
    """Persist the normalized table and complete its manifest."""
    output_dir = Path(output_dir)
    table = validate_long_table(table)
    table_path = output_dir / "interactions_long.parquet"
    table.to_parquet(table_path, index=False)
    manifest.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "output": {
                "table": table_path.name,
                "rows": len(table),
                "sha256": sha256_file(table_path),
                "schema_version": LONG_TABLE_SCHEMA,
            },
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def fail_manifest(
    manifest: dict[str, Any],
    output_dir: str | Path,
    exc: BaseException,
    *,
    started: float,
) -> None:
    """Persist an explicit failure state before re-raising an adapter error."""
    manifest.update(
        {
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
    )
    write_json(Path(output_dir) / "manifest.json", manifest)
