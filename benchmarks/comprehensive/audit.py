"""Validate the comprehensive benchmark registry and audit local readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "crychic-comprehensive-benchmark-audit-v1"
RESOURCE_MODES = frozenset({"H-common", "H-covered", "native"})

REGISTRY_COLUMNS = {
    "datasets": (
        "dataset_id",
        "title",
        "species",
        "design",
        "gold_tier",
        "role",
        "tracks",
        "n_subjects",
        "n_samples",
        "n_contexts",
        "n_cells",
        "local_paths",
        "access",
        "status",
        "reason",
    ),
    "methods": (
        "method_id",
        "display_name",
        "family",
        "tracks",
        "estimand",
        "replicate_unit",
        "multigroup_support",
        "resource_modes",
        "score_semantics",
        "score_direction",
        "adapter",
        "environment",
        "implementation_status",
        "historical_exact",
        "reason",
    ),
    "metrics": (
        "metric_id",
        "track",
        "gold_tiers",
        "endpoint_level",
        "primary",
        "direction",
        "requires_calibrated",
        "requires_truth",
        "aggregation",
        "definition",
    ),
    "panels": (
        "panel_id",
        "phase",
        "role",
        "track",
        "datasets",
        "methods",
        "resource_modes",
        "primary_metrics",
        "secondary_metrics",
        "minimum_coverage",
        "locked_after",
        "notes",
    ),
    "literature": (
        "paper_id",
        "year",
        "title",
        "doi",
        "benchmark_role",
        "canonical_url",
        "local_path",
        "license",
        "status",
    ),
    "vendor_sources": (
        "source_id",
        "canonical_url",
        "commit",
        "ref",
        "version",
        "license",
        "license_file",
        "local_path",
        "purpose",
        "status",
        "retrieved_at",
    ),
    "execution_profiles": (
        "profile_id",
        "cpu_threads",
        "parallel_jobs",
        "gpu_policy",
        "soft_memory_fraction",
        "hard_memory_fraction",
        "rss_scope",
        "timing_scope",
        "purpose",
    ),
    "simulation_scenarios": (
        "scenario_id",
        "design",
        "signal",
        "nuisance",
        "estimable",
        "primary_endpoints",
        "truth_visibility",
        "replicates",
        "notes",
    ),
    "contrasts": (
        "contrast_id",
        "dataset_id",
        "contrast_type",
        "expression",
        "inferential_unit",
        "block_key",
        "role",
        "methods_scope",
        "status",
        "notes",
    ),
}

ID_COLUMNS = {
    "datasets": "dataset_id",
    "methods": "method_id",
    "metrics": "metric_id",
    "panels": "panel_id",
    "literature": "paper_id",
    "vendor_sources": "source_id",
    "execution_profiles": "profile_id",
    "simulation_scenarios": "scenario_id",
    "contrasts": "contrast_id",
}


class RegistryError(ValueError):
    """Raised when the frozen benchmark registry is internally inconsistent."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def split_values(value: str) -> tuple[str, ...]:
    """Split a semicolon-delimited registry cell and reject duplicate values."""
    values = tuple(item.strip() for item in value.split(";") if item.strip())
    duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
    if duplicates:
        raise RegistryError(f"duplicate semicolon values: {duplicates}")
    return values


def _read_registry(path: Path, expected: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise RegistryError(f"registry file does not exist: {path.name}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        observed = tuple(reader.fieldnames or ())
        if observed != tuple(expected):
            raise RegistryError(
                f"{path.name} columns differ: expected {tuple(expected)}, "
                f"observed {observed}"
            )
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            if None in raw:
                raise RegistryError(f"{path.name}:{line_number} has extra fields")
            row = {key: (value or "").strip() for key, value in raw.items()}
            rows.append(row)
    return rows


def _index_unique(
    name: str, rows: Sequence[dict[str, str]], id_column: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        identifier = row[id_column]
        if not identifier:
            raise RegistryError(f"{name}.tsv:{line_number} has an empty {id_column}")
        if identifier in result:
            raise RegistryError(f"{name}.tsv has duplicate {id_column}={identifier!r}")
        result[identifier] = row
    return result


def load_and_validate_registries(
    registry_dir: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Load registry TSVs and validate references and declared semantics."""
    root = Path(registry_dir)
    registries: dict[str, dict[str, dict[str, str]]] = {}
    for name, columns in REGISTRY_COLUMNS.items():
        rows = _read_registry(root / f"{name}.tsv", columns)
        registries[name] = _index_unique(name, rows, ID_COLUMNS[name])

    datasets = registries["datasets"]
    methods = registries["methods"]
    metrics = registries["metrics"]

    for dataset_id, row in datasets.items():
        if row["status"] not in {
            "ready",
            "legacy_ready",
            "planned",
            "partial",
            "blocked",
            "descriptive",
        }:
            raise RegistryError(
                f"dataset {dataset_id!r} has unsupported status {row['status']!r}"
            )
        if not split_values(row["tracks"]):
            raise RegistryError(f"dataset {dataset_id!r} declares no tracks")

    for method_id, row in methods.items():
        if row["implementation_status"] not in {"runnable", "reusable", "planned"}:
            raise RegistryError(
                f"method {method_id!r} has unsupported implementation_status "
                f"{row['implementation_status']!r}"
            )
        modes = set(split_values(row["resource_modes"]))
        unknown_modes = modes.difference(RESOURCE_MODES)
        if unknown_modes:
            raise RegistryError(
                f"method {method_id!r} has unsupported resource modes "
                f"{sorted(unknown_modes)}"
            )
        if row["score_direction"] not in {"higher", "lower"}:
            raise RegistryError(
                f"method {method_id!r} has invalid score_direction "
                f"{row['score_direction']!r}"
            )

    for metric_id, row in metrics.items():
        for boolean_field in ("primary", "requires_calibrated"):
            if row[boolean_field] not in {"true", "false"}:
                raise RegistryError(
                    f"metric {metric_id!r} has invalid {boolean_field}="
                    f"{row[boolean_field]!r}"
                )

    for source_id, row in registries["vendor_sources"].items():
        if row["status"] not in {"pinned", "planned"}:
            raise RegistryError(
                f"vendor source {source_id!r} has unsupported status {row['status']!r}"
            )
        if row["status"] == "pinned" and (
            len(row["commit"]) != 40
            or any(character not in "0123456789abcdef" for character in row["commit"])
        ):
            raise RegistryError(
                f"vendor source {source_id!r} must pin a full lowercase Git commit"
            )
        if Path(row["local_path"]).is_absolute():
            raise RegistryError(
                f"vendor source {source_id!r} local_path must be workspace-relative"
            )

    for profile_id, row in registries["execution_profiles"].items():
        try:
            soft = float(row["soft_memory_fraction"])
            hard = float(row["hard_memory_fraction"])
        except ValueError as error:
            raise RegistryError(
                f"execution profile {profile_id!r} has non-numeric memory limits"
            ) from error
        if not 0 < soft < hard <= 0.80:
            raise RegistryError(
                f"execution profile {profile_id!r} must satisfy "
                "0 < soft_memory_fraction < hard_memory_fraction <= 0.80"
            )

    for scenario_id, row in registries["simulation_scenarios"].items():
        if row["estimable"] not in {"true", "false"}:
            raise RegistryError(
                f"simulation scenario {scenario_id!r} has invalid estimable="
                f"{row['estimable']!r}"
            )
        if row["truth_visibility"] not in {
            "hidden_after_freeze",
            "negative_control",
        }:
            raise RegistryError(
                f"simulation scenario {scenario_id!r} has invalid truth_visibility"
            )
        endpoint_ids = split_values(row["primary_endpoints"])
        _require_references(
            scenario_id,
            "metric",
            endpoint_ids,
            metrics,
        )
        try:
            replicates = int(row["replicates"])
        except ValueError as error:
            raise RegistryError(
                f"simulation scenario {scenario_id!r} has invalid replicates"
            ) from error
        if replicates < 1:
            raise RegistryError(
                f"simulation scenario {scenario_id!r} must have replicates >= 1"
            )

    for contrast_id, row in registries["contrasts"].items():
        _require_references(
            contrast_id,
            "dataset",
            (row["dataset_id"],),
            datasets,
        )
        _require_references(
            contrast_id,
            "method",
            split_values(row["methods_scope"]),
            methods,
        )
        if row["contrast_type"] not in {
            "omnibus",
            "pairwise",
            "one_vs_rest",
            "interaction",
        }:
            raise RegistryError(
                f"contrast {contrast_id!r} has invalid contrast_type="
                f"{row['contrast_type']!r}"
            )
        if row["status"] not in {"frozen", "planned_validation"}:
            raise RegistryError(
                f"contrast {contrast_id!r} has invalid status={row['status']!r}"
            )

    for panel_id, row in registries["panels"].items():
        dataset_ids = split_values(row["datasets"])
        method_ids = split_values(row["methods"])
        resource_modes = set(split_values(row["resource_modes"]))
        primary_metrics = split_values(row["primary_metrics"])
        secondary_metrics = split_values(row["secondary_metrics"])
        _require_references(panel_id, "dataset", dataset_ids, datasets)
        _require_references(panel_id, "method", method_ids, methods)
        _require_references(
            panel_id,
            "metric",
            (*primary_metrics, *secondary_metrics),
            metrics,
        )
        unknown_modes = resource_modes.difference(RESOURCE_MODES)
        if unknown_modes:
            raise RegistryError(
                f"panel {panel_id!r} has unsupported resource modes "
                f"{sorted(unknown_modes)}"
            )
        if not primary_metrics:
            raise RegistryError(f"panel {panel_id!r} has no primary metric")
        wrong_track = [
            metric_id
            for metric_id in primary_metrics
            if metrics[metric_id]["track"] != row["track"]
        ]
        if wrong_track:
            raise RegistryError(
                f"panel {panel_id!r} primary metrics do not match track "
                f"{row['track']!r}: {wrong_track}"
            )
        try:
            coverage = float(row["minimum_coverage"])
        except ValueError as error:
            raise RegistryError(
                f"panel {panel_id!r} has non-numeric minimum_coverage"
            ) from error
        if not 0.0 <= coverage <= 1.0:
            raise RegistryError(
                f"panel {panel_id!r} minimum_coverage must be between 0 and 1"
            )
    return registries


def _require_references(
    panel_id: str,
    kind: str,
    identifiers: Iterable[str],
    registry: Mapping[str, Mapping[str, str]],
) -> None:
    missing = sorted(set(identifiers).difference(registry))
    if missing:
        raise RegistryError(
            f"panel {panel_id!r} references unknown {kind} IDs: {missing}"
        )


def _scan_directory(path: Path, limit: int = 10_000) -> dict[str, Any]:
    file_count = 0
    nonempty_count = 0
    total_bytes = 0
    aria2_count = 0
    truncated = False
    for directory, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".aria2"):
                aria2_count += 1
            if file_count >= limit:
                truncated = True
                continue
            file_path = Path(directory) / filename
            file_count += 1
            if not filename.endswith(".aria2"):
                try:
                    size = file_path.stat().st_size
                except OSError:
                    size = 0
                total_bytes += size
                nonempty_count += int(size > 0)
    if aria2_count:
        state = "partial"
    elif nonempty_count:
        state = "complete"
    else:
        state = "empty"
    return {
        "asset_state": state,
        "exists": "true",
        "kind": "directory",
        "bytes_scanned": str(total_bytes),
        "files_scanned": str(file_count),
        "aria2_sidecars": str(aria2_count),
        "scan_truncated": str(truncated).lower(),
    }


def inspect_asset(workspace_root: Path, relative_path: str) -> dict[str, str]:
    """Inspect one declared data asset without hashing patient-scale matrices."""
    declared = Path(relative_path)
    if declared.is_absolute():
        raise RegistryError(f"local_path must be workspace-relative: {relative_path}")
    path = workspace_root / declared
    sidecar = Path(f"{path}.aria2")
    base = {"local_path": relative_path}
    if not path.exists():
        state = "partial" if sidecar.exists() else "missing"
        return {
            **base,
            "asset_state": state,
            "exists": "false",
            "kind": "missing",
            "bytes_scanned": "0",
            "files_scanned": "0",
            "aria2_sidecars": str(int(sidecar.exists())),
            "scan_truncated": "false",
        }
    if path.is_dir():
        return {**base, **_scan_directory(path)}
    size = path.stat().st_size
    state = "partial" if sidecar.exists() else ("complete" if size else "empty")
    return {
        **base,
        "asset_state": state,
        "exists": "true",
        "kind": "file",
        "bytes_scanned": str(size),
        "files_scanned": "1",
        "aria2_sidecars": str(int(sidecar.exists())),
        "scan_truncated": "false",
    }


def _dataset_asset_state(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return "undeclared"
    states = {row["asset_state"] for row in rows}
    if "partial" in states:
        return "partial"
    if states.intersection({"missing", "empty"}):
        return "unavailable"
    return "complete"


def build_asset_audit(
    datasets: Mapping[str, Mapping[str, str]], workspace_root: Path
) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows: list[dict[str, str]] = []
    summaries: dict[str, str] = {}
    for dataset_id, dataset in datasets.items():
        dataset_rows = []
        for local_path in split_values(dataset["local_paths"]):
            row = {
                "dataset_id": dataset_id,
                "declared_status": dataset["status"],
                **inspect_asset(workspace_root, local_path),
            }
            rows.append(row)
            dataset_rows.append(row)
        summaries[dataset_id] = _dataset_asset_state(dataset_rows)
    return rows, summaries


def _conda_environments() -> set[str]:
    executable = shutil.which("conda") or shutil.which("mamba")
    if executable is None:
        return set()
    try:
        result = subprocess.run(
            [executable, "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return set()
    paths = {str(Path(item).resolve()) for item in payload.get("envs", [])}
    paths.update(Path(item).name for item in payload.get("envs", []))
    return paths


def _resolve_environment(
    value: str, *, repo_root: Path, workspace_root: Path, conda_envs: set[str]
) -> tuple[str, str]:
    if value.startswith("planned:"):
        return "planned", value.removeprefix("planned:")
    if value.startswith("conda:"):
        name = value.removeprefix("conda:")
        return ("available", name) if name in conda_envs else ("missing", name)
    candidates = (repo_root / value, workspace_root / value)
    for candidate in candidates:
        if candidate.exists():
            return "available", candidate.relative_to(candidate.parents[1]).as_posix()
    return "missing", value


def _resolve_adapter(value: str, repo_root: Path) -> tuple[str, str]:
    if value.startswith("planned:"):
        return "planned", value.removeprefix("planned:")
    path = repo_root / value
    return ("available", value) if path.is_file() else ("missing", value)


def build_method_audit(
    methods: Mapping[str, Mapping[str, str]],
    *,
    repo_root: Path,
    workspace_root: Path,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    conda_envs = _conda_environments()
    rows = []
    summaries: dict[str, str] = {}
    for method_id, method in methods.items():
        adapter_state, adapter_target = _resolve_adapter(method["adapter"], repo_root)
        environment_state, environment_target = _resolve_environment(
            method["environment"],
            repo_root=repo_root,
            workspace_root=workspace_root,
            conda_envs=conda_envs,
        )
        implementation = method["implementation_status"]
        if implementation == "planned":
            readiness = "planned"
        elif implementation == "reusable":
            readiness = "reusable"
        elif adapter_state == "available" and environment_state == "available":
            readiness = "runnable"
        else:
            readiness = "unavailable"
        rows.append(
            {
                "method_id": method_id,
                "implementation_status": implementation,
                "readiness": readiness,
                "adapter_state": adapter_state,
                "adapter_target": adapter_target,
                "environment_state": environment_state,
                "environment_target": environment_target,
                "historical_exact": method["historical_exact"],
                "reason": method["reason"],
            }
        )
        summaries[method_id] = readiness
    return rows, summaries


def build_literature_audit(
    literature: Mapping[str, Mapping[str, str]], workspace_root: Path
) -> list[dict[str, str]]:
    rows = []
    for paper_id, paper in literature.items():
        asset = inspect_asset(workspace_root, paper["local_path"])
        path = workspace_root / paper["local_path"]
        checksum = (
            sha256_file(path)
            if asset["asset_state"] == "complete" and path.is_file()
            else ""
        )
        rows.append(
            {
                "paper_id": paper_id,
                "year": paper["year"],
                "doi": paper["doi"],
                "declared_status": paper["status"],
                "local_path": paper["local_path"],
                "asset_state": asset["asset_state"],
                "bytes": asset["bytes_scanned"],
                "sha256": checksum,
                "license": paper["license"],
            }
        )
    return rows


def _git_value(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_vendor_audit(
    sources: Mapping[str, Mapping[str, str]], workspace_root: Path
) -> list[dict[str, str]]:
    """Verify pinned upstream commits and declared license files."""
    rows = []
    for source_id, source in sources.items():
        path = workspace_root / source["local_path"]
        observed_commit = _git_value(path, "rev-parse", "HEAD") if path.is_dir() else ""
        dirty_output = (
            _git_value(path, "status", "--porcelain") if observed_commit else ""
        )
        license_file = source["license_file"]
        license_path = path / license_file if license_file else None
        if not path.is_dir():
            audit_state = "missing"
        elif not observed_commit:
            audit_state = "not_a_git_checkout"
        elif observed_commit != source["commit"]:
            audit_state = "commit_mismatch"
        elif license_path is not None and not license_path.is_file():
            audit_state = "license_file_missing"
        elif dirty_output:
            audit_state = "dirty_checkout"
        else:
            audit_state = "verified"
        rows.append(
            {
                "source_id": source_id,
                "declared_status": source["status"],
                "audit_state": audit_state,
                "declared_commit": source["commit"],
                "observed_commit": observed_commit,
                "ref": source["ref"],
                "version": source["version"],
                "license": source["license"],
                "license_file": license_file,
                "license_sha256": (
                    sha256_file(license_path)
                    if license_path is not None and license_path.is_file()
                    else ""
                ),
                "local_path": source["local_path"],
                "dirty": str(bool(dirty_output)).lower(),
            }
        )
    return rows


def _execution_decision(
    *,
    panel: Mapping[str, str],
    dataset: Mapping[str, str],
    method: Mapping[str, str],
    resource_mode: str,
    dataset_asset_state: str,
    method_readiness: str,
) -> tuple[str, str]:
    track = panel["track"]
    if track not in split_values(dataset["tracks"]):
        return "skipped", "dataset_track_unsupported"
    if track not in split_values(method["tracks"]):
        return "skipped", "method_track_unsupported"
    if resource_mode not in split_values(method["resource_modes"]):
        return "skipped", "resource_mode_unsupported"
    if dataset["status"] == "blocked":
        return "blocked", "controlled_or_unavailable_access"
    if dataset["status"] in {"planned", "partial"}:
        return "skipped", f"dataset_{dataset['status']}"
    if dataset["status"] == "descriptive" and track == "differential_sample":
        return "not_estimable", "no_independent_subject_replicates"
    if dataset_asset_state != "complete":
        return "skipped", f"dataset_assets_{dataset_asset_state}"
    if method_readiness == "planned":
        return "skipped", "method_adapter_or_environment_planned"
    if method_readiness == "unavailable":
        return "skipped", "method_adapter_or_environment_unavailable"
    if method_readiness == "reusable":
        if panel["role"] in {"legacy", "static"}:
            return "reusable", "frozen_output_only"
        return "skipped", "method_reusable_only"
    return "eligible", "ready_for_new_run"


def build_execution_matrix(
    registries: Mapping[str, Mapping[str, Mapping[str, str]]],
    *,
    dataset_assets: Mapping[str, str],
    method_readiness: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    datasets = registries["datasets"]
    methods = registries["methods"]
    rows = []
    for panel_id, panel in registries["panels"].items():
        for dataset_id in split_values(panel["datasets"]):
            for method_id in split_values(panel["methods"]):
                for resource_mode in split_values(panel["resource_modes"]):
                    status, reason = _execution_decision(
                        panel=panel,
                        dataset=datasets[dataset_id],
                        method=methods[method_id],
                        resource_mode=resource_mode,
                        dataset_asset_state=dataset_assets[dataset_id],
                        method_readiness=method_readiness[method_id],
                    )
                    rows.append(
                        {
                            "panel_id": panel_id,
                            "phase": panel["phase"],
                            "panel_role": panel["role"],
                            "track": panel["track"],
                            "dataset_id": dataset_id,
                            "dataset_status": datasets[dataset_id]["status"],
                            "dataset_asset_state": dataset_assets[dataset_id],
                            "method_id": method_id,
                            "method_readiness": method_readiness[method_id],
                            "resource_mode": resource_mode,
                            "execution_status": status,
                            "reason_code": reason,
                            "score_as_zero": "false",
                        }
                    )

    readiness = []
    for panel_id, panel in registries["panels"].items():
        panel_rows = [row for row in rows if row["panel_id"] == panel_id]
        counts = Counter(row["execution_status"] for row in panel_rows)
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in panel_rows:
            grouped.setdefault((row["dataset_id"], row["method_id"]), []).append(row)
        pair_statuses = []
        for pair_rows in grouped.values():
            statuses = {row["execution_status"] for row in pair_rows}
            if "eligible" in statuses:
                pair_statuses.append("eligible")
            elif "reusable" in statuses:
                pair_statuses.append("reusable")
            elif "blocked" in statuses:
                pair_statuses.append("blocked")
            elif "not_estimable" in statuses:
                pair_statuses.append("not_estimable")
            else:
                pair_statuses.append("skipped")
        pair_counts = Counter(pair_statuses)
        denominator = len(pair_statuses)
        available = pair_counts["eligible"] + pair_counts["reusable"]
        readiness.append(
            {
                "panel_id": panel_id,
                "phase": panel["phase"],
                "role": panel["role"],
                "track": panel["track"],
                "matrix_rows": str(len(panel_rows)),
                "method_dataset_pairs": str(denominator),
                "eligible_new_runs": str(counts["eligible"]),
                "reusable_runs": str(counts["reusable"]),
                "skipped_runs": str(counts["skipped"]),
                "blocked_runs": str(counts["blocked"]),
                "not_estimable_runs": str(counts["not_estimable"]),
                "eligible_pairs": str(pair_counts["eligible"]),
                "reusable_pairs": str(pair_counts["reusable"]),
                "skipped_pairs": str(pair_counts["skipped"]),
                "blocked_pairs": str(pair_counts["blocked"]),
                "not_estimable_pairs": str(pair_counts["not_estimable"]),
                "readiness_fraction": (
                    f"{available / denominator:.6f}" if denominator else "0.000000"
                ),
                "minimum_coverage": panel["minimum_coverage"],
                "coverage_gate": (
                    "pass"
                    if denominator
                    and available / denominator >= float(panel["minimum_coverage"])
                    else "not_met"
                ),
            }
        )
    return rows, readiness


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _memory_total_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 3)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _atomic_install(staging: Path, output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        staging.rename(output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory exists: {output_dir}; pass --overwrite to replace it"
        )
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    output_dir.rename(backup)
    try:
        staging.rename(output_dir)
    except BaseException:
        backup.rename(output_dir)
        raise
    shutil.rmtree(backup)


def run_audit(
    *,
    registry_dir: str | Path,
    workspace_root: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the audit and atomically install tabular outputs plus a manifest."""
    registry_root = Path(registry_dir).resolve()
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    repo_root = registry_root.parents[1]
    registries = load_and_validate_registries(registry_root)
    asset_rows, dataset_assets = build_asset_audit(registries["datasets"], workspace)
    literature_rows = build_literature_audit(registries["literature"], workspace)
    vendor_rows = build_vendor_audit(registries["vendor_sources"], workspace)
    method_rows, method_readiness = build_method_audit(
        registries["methods"], repo_root=repo_root, workspace_root=workspace
    )
    execution_rows, readiness_rows = build_execution_matrix(
        registries,
        dataset_assets=dataset_assets,
        method_readiness=method_readiness,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        tables = {
            "asset_audit.tsv": asset_rows,
            "literature_audit.tsv": literature_rows,
            "vendor_audit.tsv": vendor_rows,
            "method_audit.tsv": method_rows,
            "execution_matrix.tsv": execution_rows,
            "panel_readiness.tsv": readiness_rows,
        }
        for filename, rows in tables.items():
            _write_tsv(staging / filename, rows)
        registry_hashes = {
            f"{name}.tsv": sha256_file(registry_root / f"{name}.tsv")
            for name in REGISTRY_COLUMNS
        }
        status_counts = Counter(row["execution_status"] for row in execution_rows)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "registry": {
                "path": "benchmarks/comprehensive",
                "sha256": registry_hashes,
            },
            "git": _git_metadata(repo_root),
            "hardware": {
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "memory_total_gib": _memory_total_gib(),
            },
            "counts": {name: len(rows) for name, rows in registries.items()},
            "dataset_asset_states": dict(Counter(dataset_assets.values())),
            "method_readiness": dict(Counter(method_readiness.values())),
            "vendor_states": dict(Counter(row["audit_state"] for row in vendor_rows)),
            "execution_status": dict(status_counts),
            "outputs": {
                filename: {
                    "rows": len(rows),
                    "sha256": sha256_file(staging / filename),
                }
                for filename, rows in tables.items()
            },
            "interpretation": {
                "missing_is_zero": False,
                "not_estimable_is_zero": False,
                "resource_arms_ranked_separately": True,
                "independent_primary_panel": "P04_misc_olink",
                "legacy_panels": [
                    "P06_legacy_figure3_sample",
                    "P07_legacy_figure3_pooled",
                ],
            },
        }
        (staging / "audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_install(staging, output, overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_registry = Path(__file__).resolve().parent
    parser.add_argument("--registry-dir", type=Path, default=default_registry)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=default_registry.parents[2],
        help="Parent containing the repository, benchmark_work and datasets.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_audit(
        registry_dir=args.registry_dir,
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "execution_status": manifest["execution_status"],
                "output_dir": args.output_dir.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
