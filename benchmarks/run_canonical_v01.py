"""Run the reproducible CRYCHIC v0.1 canonical-data benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from crychic import (
    AttributionSupportMethod,
    Crychic,
    CrychicConfig,
    CrychicResult,
    load_cellchat_resource,
    load_cellphonedb_resource,
    load_nichenet_target_prior,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "benchmarks" / "configs" / "canonical_v01.json"
_SHA256_LENGTH = 64


class PeakRssMonitor:
    """Sample resident memory for the current process during one dataset run."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def start(self) -> None:
        import psutil

        process = psutil.Process()

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    rss = process.memory_info().rss
                    rss += sum(
                        child.memory_info().rss
                        for child in process.children(recursive=True)
                        if child.is_running()
                    )
                    self.peak_bytes = max(self.peak_bytes, int(rss))
                except (psutil.Error, OSError):
                    pass
                self._stop.wait(self._interval_seconds)

        self._thread = threading.Thread(
            target=sample,
            name="canonical-v01-rss",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a benchmark input without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and minimally validate the versioned canonical benchmark config."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "canonical-v0.1":
        raise ValueError("benchmark config must use schema_version canonical-v0.1")
    datasets = value.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("benchmark config must declare at least one dataset")
    for name, raw_spec in datasets.items():
        if not isinstance(name, str) or not name or not isinstance(raw_spec, dict):
            raise ValueError("dataset entries must be named objects")
        required = {
            "dataset_id",
            "input",
            "input_sha256",
            "output_name",
            "input_mode",
            "lr_resource",
            "config",
            "workflow",
        }
        missing = required.difference(raw_spec)
        if missing:
            raise ValueError(f"dataset {name!r} is missing {sorted(missing)}")
        expected = raw_spec["input_sha256"]
        if not isinstance(expected, str) or len(expected) != _SHA256_LENGTH:
            raise ValueError(f"dataset {name!r} has an invalid input_sha256")
        lr_resource = raw_spec["lr_resource"]
        resources = value.get("resources")
        if not isinstance(resources, dict) or lr_resource not in resources:
            raise ValueError(f"dataset {name!r} references an unknown LR resource")
        target_prior = raw_spec.get("target_prior")
        if target_prior is not None and target_prior not in resources:
            raise ValueError(f"dataset {name!r} references an unknown target prior")
        CrychicConfig.from_dict(raw_spec["config"])
        workflow = raw_spec["workflow"]
        if not isinstance(workflow, dict):
            raise ValueError(f"dataset {name!r} workflow must be an object")
        support_method = workflow.get("downstream_attribution_support_method")
        if support_method is not None:
            try:
                AttributionSupportMethod(str(support_method))
            except ValueError as error:
                raise ValueError(
                    f"dataset {name!r} has an invalid downstream attribution "
                    "support method"
                ) from error
    return value


def resolve_path(value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve config paths relative to the repository, never the shell cwd."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), repo_root.resolve())


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analysis_lineage(
    source_sha256: str,
    transform_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit an analyzed object to its source bytes and exact transform spec."""

    payload = {
        "schema_version": "canonical-v0.1-analysis-lineage",
        "source_sha256": source_sha256,
        "transform": dict(transform_spec),
    }
    return {**payload, "digest": _canonical_digest(payload)}


def _git_metadata(repo_root: Path) -> dict[str, Any]:
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
        "branch": run("git", "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _software_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("CRYCHIC", "anndata", "numpy", "pandas", "pyarrow", "psutil"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
    }


def _timed(timings: dict[str, float], name: str, operation: Any) -> Any:
    started = time.perf_counter()
    try:
        return operation()
    finally:
        timings[name] = time.perf_counter() - started


def _dry_run_summary(plan: Any) -> dict[str, Any]:
    support = plan.support_table
    design = plan.design_audit
    return {
        "can_fit": bool(plan.can_fit),
        "config_digest": plan.config_digest,
        "context_graph": {
            "kind": plan.context_graph.kind,
            "nodes": len(plan.context_graph.nodes),
            "edges": len(plan.context_graph.edges),
        },
        "stages": [
            {
                "name": stage.name,
                "status": stage.status.value,
                "detail": stage.detail,
            }
            for stage in plan.stages
        ],
        "warnings": list(plan.warnings),
        "design": {
            "formula": design.formula,
            "status": design.status.value,
            "reason_codes": list(design.reason_codes),
            "rank": design.rank,
            "columns": design.n_columns,
            "condition_number": design.condition_number,
            "aliased_columns": list(design.aliased_columns),
            "reference_grid_rows": len(design.reference_grid),
            "registered_contrasts": plan.contrast_table.to_dict(
                orient="records"
            ),
        },
        "resources": plan.resource_table.to_dict(orient="records"),
        "support": {
            "rows": len(support),
            "response_ready_rows": (
                int(support["response_support_ready"].sum())
                if "response_support_ready" in support
                else 0
            ),
            "eligible_samples_min": (
                int(support["n_eligible_samples"].min()) if len(support) else 0
            ),
            "eligible_samples_max": (
                int(support["n_eligible_samples"].max()) if len(support) else 0
            ),
        },
    }


def _resource_metadata(bundle: Any, prior: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ligand_receptor": {
            "resource_id": bundle.resource_id,
            "version": bundle.version,
            "species": bundle.species.value,
            "gene_namespace": bundle.gene_namespace.value,
            "manifest_digest": bundle.manifest_digest,
            "license": bundle.license,
            "interactions": len(bundle.interactions),
        },
        "target_prior": None,
    }
    if prior is not None:
        result["target_prior"] = {
            "resource_id": prior.resource_id,
            "version": prior.version,
            "species": prior.species.value,
            "gene_namespace": prior.gene_namespace.value,
            "manifest_digest": prior.manifest_digest,
            "drivers": len(prior.driver_ids),
            "targets": len(prior.target_ids),
            "links": prior.nnz,
        }
    return result


def _load_lr_resource(
    database_root: Path,
    resource_spec: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Any:
    adapter = str(resource_spec.get("adapter", ""))
    manifest_path = resolve_path(str(resource_spec["manifest"]), repo_root=repo_root)
    if adapter == "cellchat":
        return load_cellchat_resource(
            database_root,
            str(resource_spec["species"]),
            manifest_path=manifest_path,
        )
    if adapter == "cellphonedb":
        return load_cellphonedb_resource(
            database_root,
            version=str(resource_spec["version"]),
            manifest_path=manifest_path,
        )
    raise ValueError(f"unsupported LR resource adapter: {adapter!r}")


def _interaction_selection(workflow: Mapping[str, Any]) -> dict[str, Any]:
    maximum = workflow.get("max_interactions")
    if maximum is None:
        return {
            "max_interactions": None,
            "resource_truncated": False,
            "selection_rule": "all pooled-supported interactions",
            "outcome_independent": True,
            "context_label_independent": True,
        }
    return {
        "max_interactions": int(maximum),
        "resource_truncated": True,
        "selection_rule": (
            "descending sqrt(max_over_eligible_units(ligand_availability) * "
            "max_over_eligible_units(receptor_availability)); ties by "
            "interaction_id ascending"
        ),
        "outcome_independent": True,
        "context_label_independent": True,
    }


def _ordered_text_digest(values: Sequence[object]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _benchmark_manifest(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = metrics.get("result")
    return {
        "schema_version": "canonical-v0.1-benchmark-manifest",
        "dataset": metrics["dataset"],
        "dataset_id": metrics["dataset_id"],
        "status": metrics["status"],
        "benchmark_scope": metrics["benchmark_scope"],
        "benchmark_config": metrics["benchmark_config"],
        "source_input": metrics["input"],
        "analysis_lineage": metrics.get("analysis_lineage"),
        "crychic_config": metrics.get("config"),
        "effective_workflow_parameters": metrics.get("effective_workflow_parameters"),
        "interaction_selection": metrics.get("interaction_selection"),
        "threading": metrics["threading"],
        "memory_measurement": metrics["memory_measurement"],
        "resources": metrics.get("resources"),
        "dry_run": metrics.get("dry_run"),
        "result": result,
        "failure": metrics.get("failure"),
    }


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {path}; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def run_dataset(
    name: str,
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    database_root: Path,
    output_root: Path,
    resources_config: Mapping[str, Any],
    benchmark_config_path: Path,
    benchmark_config_sha256: str,
    input_override: Path | None = None,
    expected_sha256_override: str | None = None,
    min_cells_override: int | None = None,
    max_interactions_override: int | None = None,
    seed_override: int | None = None,
    blas_threads: int = 8,
    peak_rss_scope: str = "unspecified_process_scope",
    dry_run_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one configured dataset and always leave a compact metrics record."""

    dataset_output = output_root / str(spec["output_name"])
    _prepare_output(dataset_output, overwrite=overwrite)
    metrics_path = dataset_output / "metrics.json"
    benchmark_manifest_path = dataset_output / "benchmark_manifest.json"
    result_path = dataset_output / "result"
    input_path = (
        input_override.resolve()
        if input_override is not None
        else resolve_path(str(spec["input"]), repo_root=repo_root)
    )
    expected_sha256 = expected_sha256_override or str(spec["input_sha256"])
    timings: dict[str, float] = {}
    monitor = PeakRssMonitor()
    started = time.perf_counter()
    monitor.start()
    code_metadata = _git_metadata(repo_root)
    software_metadata = _software_metadata()
    metrics: dict[str, Any] = {
        "schema_version": "canonical-v0.1-metrics",
        "dataset": name,
        "dataset_id": str(spec["dataset_id"]),
        "benchmark_scope": str(spec.get("benchmark_scope", "v0_1_smoke")),
        "status": "running",
        "input_mode": str(spec["input_mode"]),
        "input": {
            "path": _display_path(input_path, repo_root),
            "expected_sha256": expected_sha256,
        },
        "output": _display_path(dataset_output, repo_root),
        "benchmark_config": {
            "path": _display_path(benchmark_config_path, repo_root),
            "sha256": benchmark_config_sha256,
        },
        "formal_inference": {
            "enabled": False,
            "p_values": None,
            "q_values": None,
            "comm_probability": None,
            "reason_code": "v0_1_inferential_disabled",
        },
        "code": code_metadata,
        "environment": software_metadata,
        "threading": {
            "threadpool_limit": blas_threads,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "memory_measurement": {
            "peak_rss_scope": peak_rss_scope,
            "isolated_process": peak_rss_scope
            in {
                "first_dataset_fresh_process",
                "standalone_isolated_process",
            },
        },
    }
    adata: ad.AnnData | None = None
    try:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        actual_sha256 = _timed(timings, "input_sha256", lambda: sha256_file(input_path))
        metrics["input"].update(
            {
                "sha256": actual_sha256,
                "bytes": input_path.stat().st_size,
                "checksum_verified": actual_sha256 == expected_sha256,
            }
        )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"input checksum mismatch for {name}: expected {expected_sha256}, "
                f"observed {actual_sha256}"
            )

        lr_resource_name = str(spec["lr_resource"])
        lr_resource_spec = resources_config[lr_resource_name]
        bundle = _timed(
            timings,
            "load_lr_resource",
            lambda: _load_lr_resource(
                database_root,
                lr_resource_spec,
                repo_root=repo_root,
            ),
        )
        prior = None
        target_name = spec.get("target_prior")
        if target_name is not None:
            prior_spec = resources_config[str(target_name)]
            prior = _timed(
                timings,
                "load_target_prior",
                lambda: load_nichenet_target_prior(
                    database_root,
                    release=str(prior_spec["release"]),
                    manifest_path=resolve_path(
                        str(prior_spec["manifest"]), repo_root=repo_root
                    ),
                ),
            )
        metrics["resources"] = _resource_metadata(bundle, prior)
        metrics["resources"]["ligand_receptor"]["config_key"] = lr_resource_name
        metrics["resources"]["ligand_receptor"]["adapter"] = str(
            lr_resource_spec["adapter"]
        )

        adata = _timed(timings, "load_input", lambda: ad.read_h5ad(input_path))
        original_shape = [int(adata.n_obs), int(adata.n_vars)]
        config_payload = dict(spec["config"])
        if seed_override is not None:
            config_payload["random_seed"] = seed_override
        config = CrychicConfig.from_dict(config_payload)
        included = spec.get("include_cell_types")
        transform_spec: dict[str, Any]
        if included is not None:
            include_values = tuple(map(str, included))
            observed = set(map(str, adata.obs[config.cell_type_key].unique()))
            missing_types = set(include_values).difference(observed)
            if missing_types:
                raise ValueError(
                    f"configured cell types absent from {name}: {sorted(missing_types)}"
                )
            mask = adata.obs[config.cell_type_key].astype(str).isin(include_values)
            adata = _timed(timings, "subset_input", lambda: adata[mask].copy())
            transform_spec = {
                "operation": "obs_membership_subset",
                "field": config.cell_type_key,
                "include_values": list(include_values),
                "membership_semantics": "string value is in include_values",
                "preserve_variable_axis": True,
                "preserve_observation_order": True,
                "materialization": "AnnData.copy",
                "selected_observations": int(adata.n_obs),
                "selected_observation_ids_sha256": _ordered_text_digest(
                    list(adata.obs_names)
                ),
            }
        else:
            transform_spec = {
                "operation": "identity",
                "preserve_observation_axis": True,
                "preserve_variable_axis": True,
            }
        lineage = analysis_lineage(actual_sha256, transform_spec)
        metrics["analysis_lineage"] = lineage

        metrics["input"].update(
            {
                "original_shape": original_shape,
                "analysis_shape": [int(adata.n_obs), int(adata.n_vars)],
                "n_samples": int(adata.obs[config.sample_key].nunique()),
                "n_subjects": int(adata.obs[config.subject_key].nunique()),
                "n_contexts": int(
                    adata.obs[list(config.context_keys)].drop_duplicates().shape[0]
                ),
                "n_cell_types": int(adata.obs[config.cell_type_key].nunique()),
                "included_cell_types": None if included is None else list(included),
            }
        )
        metrics["config"] = {
            "digest": config.digest,
            "random_seed": config.random_seed,
            "normalized_only": config.normalized_only,
            "context_keys": list(config.context_keys),
        }
        model = Crychic(config, resource_bundle=bundle, target_prior=prior)
        workflow = dict(spec["workflow"])
        if min_cells_override is not None:
            workflow["min_cells"] = min_cells_override
        if max_interactions_override is not None:
            workflow["max_interactions"] = max_interactions_override
        metrics["effective_workflow_parameters"] = workflow
        metrics["interaction_selection"] = _interaction_selection(workflow)
        dry_keys = {
            "min_cells",
            "min_samples_per_context",
            "min_subjects_per_context",
        }
        dry_kwargs = {key: workflow[key] for key in dry_keys if key in workflow}
        plan = _timed(
            timings,
            "dry_run",
            lambda: model.dry_run(adata, **dry_kwargs),
        )
        metrics["dry_run"] = _dry_run_summary(plan)
        if not plan.can_fit:
            raise RuntimeError(f"dry-run blocked fitting for {name}")
        if dry_run_only:
            metrics["status"] = "dry_run_complete"
        else:
            from threadpoolctl import threadpool_info, threadpool_limits

            def fit_with_thread_limit() -> CrychicResult | Any:
                with threadpool_limits(limits=blas_threads):
                    metrics["threading"]["libraries_during_limit"] = threadpool_info()
                    return model.fit(
                        adata,
                        output_dir=result_path,
                        input_digest=lineage["digest"],
                        git_commit=code_metadata["commit"],
                        git_dirty=bool(code_metadata["dirty"]),
                        package_version=(
                            software_metadata["packages"]["CRYCHIC"] or "0.0.0"
                        ),
                        **workflow,
                    )

            fitted = _timed(
                timings,
                "fit_and_persist",
                fit_with_thread_limit,
            )
            if not isinstance(fitted, CrychicResult):
                raise TypeError("persisted fit did not return CrychicResult")
            table_rows = {
                table: int(record["rows"])
                for table, record in fitted.manifest["tables"].items()
            }
            metrics["result"] = {
                "path": _display_path(fitted.path, repo_root),
                "run_id": fitted.manifest["run_id"],
                "mode": fitted.manifest["mode"],
                "table_rows": table_rows,
                "stages": list(fitted.manifest["stages"]),
                "warnings": list(fitted.manifest["warnings"]),
                "input_digest": fitted.manifest["input_digest"],
                "resource_digests": dict(fitted.manifest["resource_digests"]),
                "workflow_parameters": dict(fitted.manifest["workflow_parameters"]),
                "provenance": {
                    "package_version": fitted.provenance["package_version"],
                    "git_commit": fitted.provenance["git_commit"],
                    "git_dirty": fitted.provenance["git_dirty"],
                    "input_digest": fitted.provenance["input_digest"],
                },
            }
            if fitted.provenance["input_digest"] != lineage["digest"]:
                raise RuntimeError("result provenance lost the analysis lineage digest")
            metrics["status"] = "complete"
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        monitor.stop()
        metrics["timing_seconds"] = {
            **{key: round(value, 6) for key, value in timings.items()},
            "wall": round(time.perf_counter() - started, 6),
        }
        metrics["peak_rss_mb"] = round(monitor.peak_bytes / (1024**2), 3)
        metrics["benchmark_manifest"] = "benchmark_manifest.json"
        _write_json(metrics_path, metrics)
        _write_json(benchmark_manifest_path, _benchmark_manifest(metrics))
        if adata is not None:
            del adata
        gc.collect()
    return metrics


def _parse_named_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} values must use DATASET=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in result:
            raise ValueError(f"invalid or duplicate {option} value: {value!r}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def _parse_named_hashes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--expected-sha256 values must use DATASET=SHA256")
        name, digest = value.split("=", 1)
        if (
            not name
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid --expected-sha256 value: {value!r}")
        result[name] = digest
    return result


def _matching_metrics(
    datasets: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    benchmark_config_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, spec in datasets.items():
        path = output_root / str(spec["output_name"]) / "metrics.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("benchmark_config", {}).get("sha256") != (
            benchmark_config_sha256
        ):
            continue
        if record.get("dataset") != name:
            continue
        records.append(record)
    return records


def _summary_completion(
    completed_names: set[str],
    *,
    selected: Sequence[str],
    configured: Sequence[str],
) -> dict[str, Any]:
    """Separate the current invocation result from config-wide completeness."""
    requested = set(selected)
    configured_names = set(configured)
    return {
        "status": "complete" if requested.issubset(completed_names) else "failed",
        "configured_datasets_status": (
            "complete"
            if configured_names.issubset(completed_names)
            else "incomplete"
        ),
        "configured_datasets_completed": sorted(
            configured_names.intersection(completed_names)
        ),
        "configured_datasets_missing": sorted(
            configured_names.difference(completed_names)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="dataset key; repeat to run several (default: all)",
    )
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="NAME=SHA256",
    )
    parser.add_argument("--min-cells", type=int)
    parser.add_argument("--max-interactions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=8,
        help="maximum native BLAS/OpenMP threads during fit (default: 8)",
    )
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.blas_threads < 1:
        raise ValueError("--blas-threads must be positive")
    config_path = args.config.expanduser().resolve()
    benchmark_config_sha256 = sha256_file(config_path)
    config = load_benchmark_config(config_path)
    datasets: Mapping[str, Mapping[str, Any]] = config["datasets"]
    if args.list_datasets:
        for name in datasets:
            print(name)
        return 0
    selected = list(dict.fromkeys(args.dataset or datasets.keys()))
    unknown = set(selected).difference(datasets)
    if unknown:
        raise ValueError(f"unknown dataset(s): {sorted(unknown)}")
    input_overrides = _parse_named_paths(args.input, "--input")
    hash_overrides = _parse_named_hashes(args.expected_sha256)
    unknown_overrides = set(input_overrides).union(hash_overrides).difference(datasets)
    if unknown_overrides:
        raise ValueError(
            f"override references unknown dataset(s): {sorted(unknown_overrides)}"
        )

    database_root = (
        args.database_root.expanduser().resolve()
        if args.database_root is not None
        else resolve_path(config["database_root"])
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else resolve_path(config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    all_started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for index, name in enumerate(selected):
        print(f"[{name}] start", flush=True)
        record = run_dataset(
            name,
            datasets[name],
            repo_root=REPO_ROOT,
            database_root=database_root,
            output_root=output_root,
            resources_config=config["resources"],
            benchmark_config_path=config_path,
            benchmark_config_sha256=benchmark_config_sha256,
            input_override=input_overrides.get(name),
            expected_sha256_override=hash_overrides.get(name),
            min_cells_override=args.min_cells,
            max_interactions_override=args.max_interactions,
            seed_override=args.seed,
            blas_threads=args.blas_threads,
            peak_rss_scope=(
                "standalone_isolated_process"
                if len(selected) == 1
                else "first_dataset_fresh_process"
                if index == 0
                else "sequential_process_upper_bound"
            ),
            dry_run_only=args.dry_run_only,
            overwrite=args.overwrite,
        )
        records.append(record)
        print(
            f"[{name}] {record['status']} "
            f"({record['timing_seconds']['wall']:.2f}s, "
            f"{record['peak_rss_mb']:.1f} MiB peak)",
            flush=True,
        )
    current_records = _matching_metrics(datasets, output_root, benchmark_config_sha256)
    completed_names = {
        record["dataset"]
        for record in current_records
        if record["status"] in {"complete", "dry_run_complete"}
    }
    completion = _summary_completion(
        completed_names,
        selected=selected,
        configured=tuple(datasets),
    )
    summary = {
        "schema_version": "canonical-v0.1-summary",
        "benchmark_config": {
            "path": _display_path(config_path, REPO_ROOT),
            "sha256": benchmark_config_sha256,
        },
        **completion,
        "requested_datasets": selected,
        "datasets": [
            {
                "name": record["dataset"],
                "status": record["status"],
                "metrics": f"{datasets[record['dataset']]['output_name']}/metrics.json",
                "wall_seconds": record["timing_seconds"]["wall"],
                "peak_rss_mb": record["peak_rss_mb"],
                "peak_rss_scope": record.get("memory_measurement", {}).get(
                    "peak_rss_scope"
                ),
                "table_rows": record.get("result", {}).get("table_rows"),
            }
            for record in current_records
        ],
        "invocation_wall_seconds": round(time.perf_counter() - all_started, 6),
        "aggregate_dataset_wall_seconds": round(
            sum(record["timing_seconds"]["wall"] for record in current_records),
            6,
        ),
    }
    _write_json(output_root / "summary.json", summary)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
