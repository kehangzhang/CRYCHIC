"""Generate the reproducible multi-condition CCC benchmark report.

The generator is deliberately a presentation layer. It consumes frozen adapter
outputs, metric summaries, manifests, and supportive-biology annotations; it
does not recompute method scores or invent values for unavailable endpoints.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

REPORT_SCHEMA_VERSION = "multicondition-report.v1"
INPUT_SCHEMA_VERSION = "multicondition-report-inputs.v1"

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PINK = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#222222"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT_GRAY = "#F2F2F2"
PALETTE = (BLUE, VERMILLION, GREEN, ORANGE, PINK, SKY, BLACK)

ALLOWED_SYNTHETIC_SCOPES = {
    "simulation",
    "synthetic",
    "perturbation_with_known_truth",
}
OBSERVED_STATUSES = {"observed", "complete", "ok", "supported"}
BIOLOGY_SUPPORT_STATUSES = {
    "supported",
    "partial",
    "discordant",
    "not_covered",
    "not_estimable",
    "not_evaluated",
}

METRIC_FILENAMES: dict[str, tuple[str, ...]] = {
    "dataset_design": ("dataset_design.tsv", "dataset_design.csv"),
    "coverage": ("coverage_summary.tsv", "score_coverage_summary.tsv"),
    "loso_primary": (
        "loso_primary_endpoint.tsv",
        "loso_primary_summary.tsv",
    ),
    "stability": ("stability_summary.tsv", "within_context_stability.tsv"),
    "concordance": ("concordance_summary.tsv", "cross_method_concordance.tsv"),
    "performance": ("performance_summary.tsv", "run_performance_summary.tsv"),
    "biology_support": ("biology_support.tsv", "supportive_biology.tsv"),
    "simulation_truth": (
        "simulation_truth_metrics.tsv",
        "synthetic_truth_metrics.tsv",
    ),
    "iteration_comparison": (
        "iteration_comparison.tsv",
        "iteration_comparison.csv",
    ),
    "ranking_agreement": (
        "ranking_agreement_summary.tsv",
        "ranking_agreement_summary.csv",
    ),
    "ranking_top_k_curve": (
        "ranking_top_k_stability_curve.tsv",
        "ranking_top_k_stability_curve.csv",
    ),
    "ranking_intervals": (
        "bootstrap_rank_intervals.tsv",
        "bootstrap_rank_intervals.csv",
    ),
    "ranking_tiers": (
        "stable_ranking_tiers.tsv",
        "stable_ranking_tiers.csv",
    ),
}


@dataclass(frozen=True)
class ReportInputs:
    """Resolved input paths for one frozen report run."""

    final_dir: Path
    specification: Path | None
    metric_tables: Mapping[str, Path | None]
    adapter_manifests: tuple[Path, ...]
    score_tables: tuple[Path, ...]
    dataset_manifests: tuple[Path, ...]
    truth_yaml: Path | None
    supportive_biology_dataset_aliases: Mapping[str, str]
    alias_source: Path | None = None

    def paths_with_roles(self) -> list[tuple[Path, str]]:
        rows: list[tuple[Path, str]] = []
        if self.specification is not None:
            rows.append((self.specification, "report input specification"))
        rows.extend(
            (path, f"metric table: {name}")
            for name, path in self.metric_tables.items()
            if path is not None
        )
        rows.extend((path, "adapter manifest") for path in self.adapter_manifests)
        rows.extend((path, "adapter long score table") for path in self.score_tables)
        rows.extend((path, "dataset manifest") for path in self.dataset_manifests)
        if self.truth_yaml is not None:
            rows.append((self.truth_yaml, "locked supportive-biology specification"))
        if self.alias_source is not None:
            rows.append((self.alias_source, "supportive-biology dataset aliases"))
        return rows


def _native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _executable_version(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        return "not_installed"
    result = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True
    )
    value = (result.stdout or result.stderr).strip()
    return value if result.returncode == 0 and value else "unavailable"


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _git_state(repo_root: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    status_text = status.stdout if status.returncode == 0 else ""
    diff_bytes = diff.stdout if diff.returncode == 0 else b""
    return {
        "dirty": bool(status_text),
        "status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "status_entry_count": len(status_text.splitlines()),
    }


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path_list(root: Path, value: Any, *, field: str) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise ValueError(f"{field} must be a path string or list of path strings")
    paths = tuple(_resolve_path(root, item) for item in values)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{field} contains missing files: {missing}")
    return paths


def _discover_one(root: Path, names: Sequence[str]) -> Path | None:
    matches: list[Path] = []
    for name in names:
        matches.extend(path for path in root.rglob(name) if path.is_file())
    unique = sorted(set(matches))
    if len(unique) > 1:
        choices = ", ".join(str(path) for path in unique)
        raise ValueError(
            f"multiple candidate metric tables found ({choices}); "
            "select one in report_inputs.json"
        )
    return unique[0] if unique else None


def resolve_inputs(
    final_dir: Path,
    *,
    input_spec: Path | None = None,
    default_truth: Path | None = None,
) -> ReportInputs:
    """Resolve an explicit report input contract or standard-name discovery."""

    root = final_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if input_spec is None:
        candidate = root / "report_inputs.json"
    elif input_spec.is_absolute():
        candidate = input_spec
    else:
        candidate = root / input_spec
    specification = candidate.resolve() if candidate.is_file() else None
    payload: dict[str, Any] = {}
    if specification is not None:
        raw = json.loads(specification.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("report_inputs.json must contain a JSON object")
        payload = raw
        schema = payload.get("schema_version")
        if schema != INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"report input schema must be {INPUT_SCHEMA_VERSION!r}; got {schema!r}"
            )

    metric_payload = payload.get("metrics", {})
    if not isinstance(metric_payload, dict):
        raise ValueError("report_inputs.json metrics must be an object")
    metric_tables: dict[str, Path | None] = {}
    for key, filenames in METRIC_FILENAMES.items():
        explicit = metric_payload.get(key)
        if explicit is not None:
            if not isinstance(explicit, str):
                raise ValueError(f"metrics.{key} must be a path string")
            path = _resolve_path(root, explicit)
            if not path.is_file():
                raise FileNotFoundError(path)
            metric_tables[key] = path
        else:
            metric_tables[key] = _discover_one(root, filenames)

    adapter_manifests = _path_list(
        root,
        payload.get("adapter_manifests"),
        field="adapter_manifests",
    )
    if not adapter_manifests:
        adapter_manifests = tuple(sorted(root.rglob("adapter_manifest.json")))
    score_tables = _path_list(
        root,
        payload.get("score_tables"),
        field="score_tables",
    )
    if not score_tables:
        score_tables = tuple(sorted(root.rglob("*long*.parquet")))
    dataset_manifests = _path_list(
        root,
        payload.get("dataset_manifests"),
        field="dataset_manifests",
    )
    if not dataset_manifests:
        dataset_manifests = tuple(sorted(root.rglob("dataset_manifest.json")))

    truth_yaml: Path | None
    explicit_truth = payload.get("truth_yaml")
    if explicit_truth is not None:
        if not isinstance(explicit_truth, str):
            raise ValueError("truth_yaml must be a path string")
        truth_yaml = _resolve_path(root, explicit_truth)
        if not truth_yaml.is_file():
            raise FileNotFoundError(truth_yaml)
    elif default_truth is not None and default_truth.is_file():
        truth_yaml = default_truth.resolve()
    else:
        discovered_truth = sorted(root.rglob("*supportive*biology*.yaml"))
        truth_yaml = discovered_truth[0] if len(discovered_truth) == 1 else None

    alias_source: Path | None = None
    alias_payload = payload.get("supportive_biology_dataset_aliases")
    if alias_payload is None:
        finalization_manifest = root / "finalization_manifest.json"
        if finalization_manifest.is_file():
            finalization_payload = json.loads(
                finalization_manifest.read_text(encoding="utf-8")
            )
            if not isinstance(finalization_payload, dict):
                raise ValueError("finalization_manifest.json must contain an object")
            alias_payload = finalization_payload.get(
                "supportive_biology_dataset_aliases", {}
            )
            alias_source = finalization_manifest.resolve()
        else:
            alias_payload = {}
    if not isinstance(alias_payload, dict):
        raise ValueError("supportive_biology_dataset_aliases must be an object")
    biology_dataset_aliases: dict[str, str] = {}
    for locked_dataset, benchmark_dataset in alias_payload.items():
        if not isinstance(locked_dataset, str) or not isinstance(
            benchmark_dataset, str
        ):
            raise ValueError(
                "supportive_biology_dataset_aliases keys and values must be strings"
            )
        biology_dataset_aliases[locked_dataset] = benchmark_dataset

    return ReportInputs(
        final_dir=root,
        specification=specification,
        metric_tables=metric_tables,
        adapter_manifests=adapter_manifests,
        score_tables=score_tables,
        dataset_manifests=dataset_manifests,
        truth_yaml=truth_yaml,
        supportive_biology_dataset_aliases=biology_dataset_aliases,
        alias_source=alias_source,
    )


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return pd.DataFrame(value)
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return pd.DataFrame(value["records"])
    raise ValueError(f"unsupported table format: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _first_present(table: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    for name in names:
        if name in table:
            return table[name]
    return pd.Series(pd.NA, index=table.index, dtype="object")


def _numeric(table: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    return pd.to_numeric(_first_present(table, names), errors="coerce")


def _text(table: pd.DataFrame, names: Sequence[str], default: str = "") -> pd.Series:
    values = _first_present(table, names)
    return values.fillna(default).astype(str)


def _status(table: pd.DataFrame, default: str = "observed") -> pd.Series:
    values = _text(table, ("status", "endpoint_status"), default)
    return values.replace("", default)


def _ne_source(*, reason_code: str, dataset: str = "All") -> pd.DataFrame:
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": "NE",
        "method_label": "NE",
        "method_left": "NE",
        "method_right": "NE",
        "resource_mode": "not_available",
        "analysis_track": "not_available",
        "contrast": "not_available",
        "metric": "not_available",
        "truth_scope": "not_available",
        "status": "not_estimable",
        "reason_code": reason_code,
        "design_type": "not_available",
        "iteration_from": "not_available",
        "iteration_to": "not_available",
        "metric_direction": "higher",
        "change_class": "not_estimable",
        "ranking_level": "not_available",
        "item_id": "__not_estimable__",
        "item_label": "NE",
        "tier": "not_estimable",
    }
    for column in (
        "n_cells",
        "n_samples",
        "n_subjects",
        "n_contexts",
        "n_cell_types",
        "resource_coverage_fraction",
        "comparison_coverage_fraction",
        "observed_rows",
        "not_predicted_rows",
        "missing_rows",
        "cell_type_missing_rows",
        "not_estimable_rows",
        "failed_rows",
        "not_supported_rows",
        "estimate",
        "ci_lower",
        "ci_upper",
        "n_subjects_estimable",
        "effect_spearman",
        "direction_agreement",
        "shared_edges",
        "median_wall_time_seconds",
        "median_peak_rss_mb",
        "success_rate",
        "n_failed",
        "before",
        "after",
        "signed_improvement",
        "relative_improvement",
        "ranking_universe_size",
        "n_repeats_requested",
        "n_repeats_estimable",
        "k",
        "lower_rank",
        "median_rank",
        "upper_rank",
        "rank_availability_frequency",
        "top_k_frequency",
        "n_replicates_requested",
    ):
        row[column] = math.nan
    return pd.DataFrame([row])


def _method_labels(table: pd.DataFrame) -> pd.Series:
    method = _text(table, ("method", "method_id"), "unknown")
    resource_mode = _text(table, ("resource_mode",), "")
    labels = method.copy()
    has_arm = resource_mode.ne("")
    labels.loc[has_arm] = method.loc[has_arm] + " | " + resource_mode.loc[has_arm]
    return labels


def _short_dataset_labels(values: pd.Series) -> pd.Series:
    return values.replace(
        {
            "GSE144236_Ji_cSCC": "cSCC",
            "UCSC_Lerma_Martin_MS_snRNA_CA_vs_Ctrl": "MS CA vs Ctrl",
            "__cross_dataset__": "Cross-dataset",
            "Kuppe_MI_Zenodo6578047": "Kuppe MI",
            "GSE244992_Panc02_PancVAX": "Panc02",
            "synthetic_active": "Syn active",
            "synthetic_global_null": "Syn global null",
            "synthetic_abundance_only": "Syn abundance",
            "synthetic_receiver_autonomous": "Syn receiver-auto",
            "synthetic_ligand_only": "Syn ligand-only",
            "synthetic_target_only": "Syn target-only",
            "synthetic_receptor_knockout": "Syn receptor KO",
        }
    )


def _display_method_names(values: pd.Series) -> pd.Series:
    return values.replace(
        {
            "cellchat": "CellChat",
            "cellphonedb": "CellPhoneDB",
            "liana_rank_aggregate": "LIANA",
            "crychic": "CRYCHIC",
            "nichenet_prior_activity": "NicheNet-prior",
        }
    )


def _load_manifests(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = _read_json(path)
        except (json.JSONDecodeError, ValueError) as exc:
            records.append(
                {
                    "path": str(path),
                    "status": "failed",
                    "reason_code": f"invalid_manifest:{exc}",
                }
            )
            continue
        record = {
            "path": str(path),
            "dataset": payload.get("dataset_id", payload.get("dataset", "")),
            "method": payload.get("method_id", payload.get("method", "")),
            "method_version": payload.get("method_version", ""),
            "resource": payload.get("resource_id", payload.get("resource", "")),
            "status": payload.get("status", "recorded"),
            "reason_code": payload.get("reason_code", ""),
        }
        records.append(record)
    return records


def _dataset_manifest_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        input_payload = payload.get("input", {})
        if not isinstance(input_payload, dict):
            input_payload = {}
        design = payload.get("design_audit", payload.get("design", {}))
        if not isinstance(design, dict):
            design = {}
        shape = input_payload.get("analysis_shape", input_payload.get("shape", []))
        n_cells = shape[0] if isinstance(shape, list) and shape else None
        rows.append(
            {
                "dataset": payload.get(
                    "dataset_id", payload.get("dataset", path.parent.name)
                ),
                "n_cells": input_payload.get("n_cells", n_cells),
                "n_samples": input_payload.get("n_samples"),
                "n_subjects": input_payload.get("n_subjects"),
                "n_contexts": input_payload.get("n_contexts"),
                "n_cell_types": input_payload.get("n_cell_types"),
                "design_type": design.get(
                    "design_type", payload.get("design_type", "not_recorded")
                ),
                "status": design.get("status", payload.get("status", "recorded")),
                "reason_code": design.get(
                    "reason_code", payload.get("reason_code", "")
                ),
                "manifest_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _design_source(
    table: pd.DataFrame, dataset_manifests: Sequence[Path]
) -> pd.DataFrame:
    if table.empty:
        table = _dataset_manifest_rows(dataset_manifests)
    if table.empty:
        return _ne_source(reason_code="dataset_design_table_missing")
    source = pd.DataFrame(
        {
            "dataset": _text(table, ("dataset", "dataset_id"), "unknown"),
            "n_cells": _numeric(table, ("n_cells", "cells")),
            "n_samples": _numeric(table, ("n_samples", "samples")),
            "n_subjects": _numeric(table, ("n_subjects", "subjects")),
            "n_contexts": _numeric(table, ("n_contexts", "contexts")),
            "n_cell_types": _numeric(table, ("n_cell_types", "cell_types")),
            "design_type": _text(
                table, ("design_type", "design", "design_status"), "not_recorded"
            ),
            "truth_scope": _text(table, ("truth_scope",), "unknown"),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    return source.sort_values("dataset", kind="stable", ignore_index=True)


def _coverage_from_long(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        table = pd.read_parquet(path)
        required = {"status"}
        if required.difference(table.columns):
            continue
        prepared = table.copy()
        prepared["method"] = _text(prepared, ("method", "method_id"), path.stem)
        prepared["dataset"] = _text(
            prepared, ("dataset", "dataset_id"), path.parent.name
        )
        prepared["resource_mode"] = _text(prepared, ("resource_mode",), "unknown")
        frames.append(prepared)
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    grouping = ["dataset", "method", "resource_mode"]
    for keys, group in long.groupby(grouping, sort=False, observed=True):
        statuses = group["status"].fillna("missing").astype(str)
        n_rows = len(group)
        unavailable = statuses.isin({"resource_unavailable", "unsupported_resource"})
        comparable = statuses.isin({"ok", "observed", "not_returned", "not_predicted"})
        failed = statuses.isin({"method_failed", "failed"})
        dataset, method, resource_mode = keys
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "resource_mode": resource_mode,
                "resource_coverage_fraction": float((~unavailable).mean()),
                "comparison_coverage_fraction": float(comparable.mean()),
                "failed_rows": int(failed.sum()),
                "status": "observed" if comparable.any() else "not_estimable",
                "reason_code": "" if comparable.any() else "no_comparable_scores",
                "total_rows": n_rows,
            }
        )
    return pd.DataFrame(rows)


def _coverage_source(table: pd.DataFrame, score_tables: Sequence[Path]) -> pd.DataFrame:
    if table.empty:
        table = _coverage_from_long(score_tables)
    if table.empty:
        return _ne_source(reason_code="coverage_table_and_long_scores_missing")
    source = table.copy(deep=True)
    source["dataset"] = _text(source, ("dataset", "dataset_id"), "unknown")
    source["method"] = _text(source, ("method", "method_id"), "unknown")
    source["resource_mode"] = _text(source, ("resource_mode",), "unknown")
    source["analysis_track"] = _text(source, ("analysis_track",), "lr_stlr")
    source["truth_scope"] = _text(source, ("truth_scope",), "unknown")
    source["method_label"] = (
        _display_method_names(source["method"]) + " | " + source["resource_mode"]
    )
    proxy = source["analysis_track"].eq("ligand_target_program")
    source.loc[proxy, "method_label"] += " | proxy diagnostic"
    source["resource_coverage_fraction"] = _numeric(
        source,
        ("resource_coverage_fraction", "resource_coverage"),
    )
    source["comparison_coverage_fraction"] = _numeric(
        source,
        ("comparison_coverage_fraction", "median_sample_comparison_coverage"),
    )
    source["status"] = _status(source, "not_estimable")
    source["reason_code"] = _text(source, ("reason_code",), "")
    for column in (
        "observed_rows",
        "not_predicted_rows",
        "missing_rows",
        "cell_type_missing_rows",
        "not_estimable_rows",
        "failed_rows",
        "not_supported_rows",
    ):
        source[column] = _numeric(source, (column,)).fillna(0)
    for column in (
        "resource_coverage_fraction",
        "comparison_coverage_fraction",
    ):
        values = source[column].dropna()
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must lie in [0, 1] when supplied")
    return source


def _loso_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="loso_primary_endpoint_missing")
    source = pd.DataFrame(
        {
            "dataset": _text(table, ("dataset",), "unknown"),
            "method": _text(table, ("method", "method_id"), "unknown"),
            "resource_mode": _text(table, ("resource_mode",), "unknown"),
            "analysis_track": _text(table, ("analysis_track",), "lr_stlr"),
            "truth_scope": _text(table, ("truth_scope",), "unknown"),
            "endpoint_scope": _text(table, ("endpoint_scope",), "dataset"),
            "contrast": _text(table, ("contrast",), "not_recorded"),
            "rank_scope": _text(
                table, ("rank_scope",), "global_common_functional"
            ),
            "estimate": _numeric(
                table,
                ("estimate", "effect_spearman", "macro_effect_spearman"),
            ),
            "ci_lower": _numeric(table, ("ci_lower", "lower", "lower_ci")),
            "ci_upper": _numeric(table, ("ci_upper", "upper", "upper_ci")),
            "n_subjects_estimable": _numeric(
                table, ("n_subjects_estimable", "n_estimable_subjects")
            ),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    observed = source["status"].isin(OBSERVED_STATUSES)
    invalid = observed & (
        source[["estimate", "ci_lower", "ci_upper"]].isna().any(axis=1)
        | (source["ci_lower"] > source["estimate"])
        | (source["ci_upper"] < source["estimate"])
    )
    if invalid.any():
        raise ValueError("observed LOSO rows require estimate and ordered CI bounds")
    bounded = source.loc[observed, ["estimate", "ci_lower", "ci_upper"]]
    if ((bounded < -1) | (bounded > 1)).any().any():
        raise ValueError(
            "LOSO Spearman estimates and confidence bounds must be in [-1, 1]"
        )
    source = source.loc[source["truth_scope"].eq("real_data")].reset_index(drop=True)
    if source.empty:
        return _ne_source(reason_code="real_data_primary_endpoint_missing")
    return source


def _stability_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="stability_summary_missing")
    identity = pd.DataFrame(
        {
            "dataset": _text(table, ("dataset",), "unknown"),
            "method": _text(table, ("method", "method_id"), "unknown"),
            "resource_mode": _text(table, ("resource_mode",), "unknown"),
            "analysis_track": _text(table, ("analysis_track",), "lr_stlr"),
            "truth_scope": _text(table, ("truth_scope",), "unknown"),
            "contrast": _text(table, ("contrast",), "not_recorded"),
            "rank_scope": _text(
                table, ("rank_scope",), "global_common_functional"
            ),
            "context": _text(table, ("context",), ""),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    if {"metric", "estimate"}.issubset(table.columns):
        identity["metric"] = table["metric"].astype(str)
        identity["estimate"] = pd.to_numeric(table["estimate"], errors="coerce")
        return identity
    metric_columns = [
        name
        for name in (
            "median_spearman",
            "median_top_k_jaccard",
            "effect_spearman",
            "direction_agreement",
        )
        if name in table
    ]
    if not metric_columns:
        return _ne_source(reason_code="stability_metric_columns_missing")
    frames: list[pd.DataFrame] = []
    for column in metric_columns:
        frame = identity.copy()
        frame["metric"] = column
        frame["estimate"] = pd.to_numeric(table[column], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _ranking_identity_source(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": _text(table, ("dataset",), "unknown"),
            "method": _text(table, ("method", "method_id"), "unknown"),
            "resource_mode": _text(table, ("resource_mode",), "unknown"),
            "analysis_track": _text(table, ("analysis_track",), "lr_stlr"),
            "truth_scope": _text(table, ("truth_scope",), "unknown"),
            "contrast": _text(table, ("contrast",), "not_recorded"),
            "rank_scope": _text(
                table, ("rank_scope",), "global_common_functional"
            ),
            "design": _text(table, ("design",), "not_available"),
            "reference": _text(table, ("reference",), "not_available"),
            "target": _text(table, ("target",), "not_available"),
            "ranking_level": _text(table, ("ranking_level",), "unknown"),
            "ranking_universe_size": _numeric(table, ("ranking_universe_size",)),
            "n_bootstrap": _numeric(table, ("n_bootstrap",)),
            "n_split_repeats": _numeric(table, ("n_split_repeats",)),
            "rank_interval_quantile_level": _numeric(
                table, ("rank_interval_quantile_level",)
            ),
            "minimum_top_k_frequency": _numeric(
                table, ("minimum_top_k_frequency",)
            ),
            "minimum_estimable_replicate_fraction": _numeric(
                table, ("minimum_estimable_replicate_fraction",)
            ),
            "minimum_subjects": _numeric(table, ("minimum_subjects",)),
            "minimum_observed_ranks": _numeric(
                table, ("minimum_observed_ranks",)
            ),
            "n_reference_subjects": _numeric(table, ("n_reference_subjects",)),
            "n_target_subjects": _numeric(table, ("n_target_subjects",)),
            "n_paired_subjects": _numeric(table, ("n_paired_subjects",)),
            "rbo_persistence": _numeric(table, ("rbo_persistence",)),
            "weighted_kendall_power": _numeric(
                table, ("weighted_kendall_power",)
            ),
            "random_seed": _numeric(table, ("random_seed",)),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )


def _validate_nichenet_rank_claims(source: pd.DataFrame) -> None:
    unsupported = (
        source["method"].str.contains("nichenet", case=False, regex=False)
        & source["ranking_level"].isin({"lr", "sender"})
        & source["status"].isin(OBSERVED_STATUSES)
    )
    if unsupported.any():
        raise ValueError(
            "NicheNet Track B cannot report observed LR or sender rankings"
        )


def _validate_rank_scope_claims(source: pd.DataFrame) -> None:
    observed = source["status"].isin(OBSERVED_STATUSES)
    allowed = {"global_common_functional", "within_receiver_macro"}
    if (observed & ~source["rank_scope"].isin(allowed)).any():
        raise ValueError("observed ranking rows require an estimable rank_scope")


def _validate_rank_protocol(source: pd.DataFrame) -> None:
    observed = source["status"].isin(OBSERVED_STATUSES)
    expected = {
        "n_bootstrap": 2000.0,
        "n_split_repeats": 200.0,
        "rank_interval_quantile_level": 0.95,
        "minimum_top_k_frequency": 0.8,
        "minimum_estimable_replicate_fraction": 0.8,
        "minimum_observed_ranks": 2.0,
        "rbo_persistence": 0.9,
        "weighted_kendall_power": 1.0,
        "random_seed": 20260712.0,
    }
    for column, value in expected.items():
        close = pd.Series(
            np.isclose(source[column].to_numpy(dtype=float), value, atol=1e-12),
            index=source.index,
        )
        invalid = observed & (source[column].isna() | ~close)
        if invalid.any():
            raise ValueError(
                f"observed ranking rows violate preregistered {column}={value:g}"
            )
    minimum_subjects = source["minimum_subjects"]
    invalid_minimum = observed & (
        minimum_subjects.isna()
        | minimum_subjects.lt(3)
        | minimum_subjects.mod(1).ne(0)
    )
    if invalid_minimum.any():
        raise ValueError(
            "observed ranking rows require an integer minimum_subjects >= 3"
        )


def _ranking_agreement_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="ranking_agreement_missing")
    source = _ranking_identity_source(table)
    source["metric"] = _text(table, ("metric",), "unknown")
    source["estimate"] = _numeric(table, ("estimate",))
    source["envelope_lower"] = _numeric(table, ("envelope_lower",))
    source["envelope_upper"] = _numeric(table, ("envelope_upper",))
    source["interval_type"] = _text(table, ("interval_type",), "")
    source["n_repeats_requested"] = _numeric(table, ("n_repeats_requested",))
    source["n_repeats_estimable"] = _numeric(table, ("n_repeats_estimable",))
    observed = source["status"].isin(OBSERVED_STATUSES)
    invalid = observed & (
        source[["estimate", "envelope_lower", "envelope_upper"]]
        .isna()
        .any(axis=1)
        | source["envelope_lower"].gt(source["estimate"])
        | source["envelope_upper"].lt(source["estimate"])
        | source["interval_type"].ne("split_repeat_quantile_envelope")
    )
    if invalid.any():
        raise ValueError("observed ranking agreement requires ordered intervals")
    rbo = observed & source["metric"].eq("rank_biased_overlap")
    kendall = observed & source["metric"].eq("weighted_kendall_tau")
    rbo_values = source.loc[
        rbo, ["estimate", "envelope_lower", "envelope_upper"]
    ]
    kendall_values = source.loc[
        kendall, ["estimate", "envelope_lower", "envelope_upper"]
    ]
    if (
        ((rbo_values < 0) | (rbo_values > 1)).any().any()
        or ((kendall_values < -1) | (kendall_values > 1)).any().any()
    ):
        raise ValueError("ranking agreement estimates are outside metric bounds")
    _validate_nichenet_rank_claims(source)
    _validate_rank_scope_claims(source)
    _validate_rank_protocol(source)
    source = source.loc[source["truth_scope"].eq("real_data")].reset_index(drop=True)
    return source if not source.empty else _ne_source(
        reason_code="real_data_ranking_agreement_missing"
    )


def _ranking_curve_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="ranking_top_k_curve_missing")
    source = _ranking_identity_source(table)
    source["metric"] = _text(table, ("metric",), "top_k_jaccard")
    source["k"] = _numeric(table, ("k",))
    source["estimate"] = _numeric(table, ("estimate",))
    source["envelope_lower"] = _numeric(table, ("envelope_lower",))
    source["envelope_upper"] = _numeric(table, ("envelope_upper",))
    source["interval_type"] = _text(table, ("interval_type",), "")
    source["n_repeats_estimable"] = _numeric(table, ("n_repeats_estimable",))
    source["median_realized_k_left"] = _numeric(
        table, ("median_realized_k_left",)
    )
    source["median_realized_k_right"] = _numeric(
        table, ("median_realized_k_right",)
    )
    source["median_boundary_tie_size_left"] = _numeric(
        table, ("median_boundary_tie_size_left",)
    )
    source["median_boundary_tie_size_right"] = _numeric(
        table, ("median_boundary_tie_size_right",)
    )
    source["boundary_tie_repeat_fraction"] = _numeric(
        table, ("boundary_tie_repeat_fraction",)
    )
    observed = source["status"].isin(OBSERVED_STATUSES)
    invalid = observed & (
        source[["k", "estimate", "envelope_lower", "envelope_upper"]]
        .isna()
        .any(axis=1)
        | source["k"].lt(1)
        | source["envelope_lower"].gt(source["estimate"])
        | source["envelope_upper"].lt(source["estimate"])
        | source["interval_type"].ne("split_repeat_quantile_envelope")
    )
    bounded = source.loc[
        observed, ["estimate", "envelope_lower", "envelope_upper"]
    ]
    if invalid.any() or ((bounded < 0) | (bounded > 1)).any().any():
        raise ValueError("observed top-k stability curves are invalid")
    _validate_nichenet_rank_claims(source)
    _validate_rank_scope_claims(source)
    _validate_rank_protocol(source)
    source = source.loc[source["truth_scope"].eq("real_data")].reset_index(drop=True)
    return source if not source.empty else _ne_source(
        reason_code="real_data_ranking_top_k_curve_missing"
    )


def _ranking_intervals_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="bootstrap_rank_intervals_missing")
    source = _ranking_identity_source(table)
    source["item_id"] = _text(table, ("item_id",), "")
    source["item_label"] = _text(table, ("item_label",), "")
    source["lower_rank"] = _numeric(table, ("lower_rank",))
    source["median_rank"] = _numeric(table, ("median_rank",))
    source["upper_rank"] = _numeric(table, ("upper_rank",))
    source["rank_availability_frequency"] = _numeric(
        table, ("rank_availability_frequency",)
    )
    source["top_k_frequency"] = _numeric(table, ("top_k_frequency",))
    source["n_replicates_requested"] = _numeric(
        table, ("n_replicates_requested",)
    )
    observed = source["status"].isin(OBSERVED_STATUSES)
    invalid = observed & (
        source[
            [
                "lower_rank",
                "median_rank",
                "upper_rank",
                "rank_availability_frequency",
                "top_k_frequency",
            ]
        ].isna().any(axis=1)
        | source["lower_rank"].lt(1)
        | source["lower_rank"].gt(source["median_rank"])
        | source["median_rank"].gt(source["upper_rank"])
        | ~source["rank_availability_frequency"].between(0, 1)
        | ~source["top_k_frequency"].between(0, 1)
    )
    if invalid.any():
        raise ValueError("observed bootstrap rank intervals are invalid")
    _validate_nichenet_rank_claims(source)
    _validate_rank_scope_claims(source)
    _validate_rank_protocol(source)
    source = source.loc[source["truth_scope"].eq("real_data")].reset_index(drop=True)
    return source if not source.empty else _ne_source(
        reason_code="real_data_bootstrap_rank_intervals_missing"
    )


def _ranking_tiers_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="stable_ranking_tiers_missing")
    source = _ranking_identity_source(table)
    source["item_id"] = _text(table, ("item_id",), "")
    source["item_label"] = _text(table, ("item_label",), "")
    source["tier"] = _text(table, ("tier",), "not_estimable")
    source["rank_availability_frequency"] = _numeric(
        table, ("rank_availability_frequency",)
    )
    source["top_k_frequency"] = _numeric(table, ("top_k_frequency",))
    source["lower_rank"] = _numeric(table, ("lower_rank",))
    source["upper_rank"] = _numeric(table, ("upper_rank",))
    allowed_tiers = {
        "stable_top_k",
        "possible_top_k",
        "stable_below_top_k",
        "unstable",
        "not_estimable",
    }
    if not set(source["tier"]).issubset(allowed_tiers):
        raise ValueError("stable ranking table contains an unsupported tier")
    _validate_nichenet_rank_claims(source)
    _validate_rank_scope_claims(source)
    _validate_rank_protocol(source)
    source = source.loc[source["truth_scope"].eq("real_data")].reset_index(drop=True)
    return source if not source.empty else _ne_source(
        reason_code="real_data_stable_ranking_tiers_missing"
    )


def _concordance_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="cross_method_concordance_missing")
    source = pd.DataFrame(
        {
            "dataset": _text(table, ("dataset",), "unknown"),
            "method_left": _text(table, ("method_left", "left_method"), "unknown-left"),
            "method_right": _text(
                table, ("method_right", "right_method"), "unknown-right"
            ),
            "resource_mode": _text(table, ("resource_mode",), "unknown"),
            "analysis_track": _text(table, ("analysis_track",), "lr_stlr"),
            "truth_scope": _text(table, ("truth_scope",), "unknown"),
            "contrast": _text(table, ("contrast",), "not_recorded"),
            "rank_scope": _text(
                table, ("rank_scope",), "global_common_functional"
            ),
            "effect_spearman": _numeric(
                table, ("effect_spearman", "spearman", "rank_correlation")
            ),
            "direction_agreement": _numeric(table, ("direction_agreement",)),
            "shared_edges": _numeric(table, ("shared_edges",)),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    observed = source["status"].isin(OBSERVED_STATUSES)
    if (observed & source["rank_scope"].ne("global_common_functional")).any():
        raise ValueError(
            "cross-method concordance requires scope-matched global functionals"
        )
    return source


def _performance_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="performance_summary_missing")
    return pd.DataFrame(
        {
            "dataset": _text(table, ("dataset",), "unknown"),
            "method": _text(table, ("method", "method_id"), "unknown"),
            "resource_mode": _text(table, ("resource_mode",), "unknown"),
            "analysis_track": _text(table, ("analysis_track",), "lr_stlr"),
            "truth_scope": _text(table, ("truth_scope",), "unknown"),
            "median_wall_time_seconds": _numeric(
                table, ("median_wall_time_seconds", "wall_time_seconds")
            ),
            "median_peak_rss_mb": _numeric(
                table, ("median_peak_rss_mb", "peak_rss_mb")
            ),
            "success_rate": _numeric(table, ("success_rate",)),
            "n_failed": _numeric(table, ("n_failed", "failed_runs")),
            "threads_minimum": _numeric(table, ("threads_minimum",)),
            "threads_maximum": _numeric(table, ("threads_maximum",)),
            "method_runtime_role": _text(
                table, ("method_runtime_role",), "method_total"
            ),
            "median_method_total_wall_time_seconds": _numeric(
                table, ("median_method_total_wall_time_seconds",)
            ),
            "median_core_fit_wall_time_seconds": _numeric(
                table, ("median_core_fit_wall_time_seconds",)
            ),
            "median_source_pipeline_total_wall_time_seconds": _numeric(
                table, ("median_source_pipeline_total_wall_time_seconds",)
            ),
            "median_adapter_readback_wall_time_seconds": _numeric(
                table, ("median_adapter_readback_wall_time_seconds",)
            ),
            "performance_component_count": _numeric(
                table, ("performance_component_count",)
            ),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )


def _truth_observations(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("supportive biology YAML must contain an object")
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("supportive biology YAML datasets must be an object")
    rows: list[dict[str, str]] = []
    for dataset, dataset_payload in datasets.items():
        if not isinstance(dataset_payload, dict):
            continue
        observations = dataset_payload.get("expected_observations", [])
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if not isinstance(observation, dict) or "id" not in observation:
                continue
            rows.append(
                {
                    "dataset": str(dataset),
                    "observation_id": str(observation["id"]),
                    "expected_direction": str(
                        observation.get("direction", "not_directional")
                    ),
                    "expected_metric": str(
                        observation.get("metric", "supportive_only")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _biology_source(
    table: pd.DataFrame,
    truth_yaml: Path | None,
    dataset_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    truth = _truth_observations(truth_yaml)
    if truth.empty:
        return _ne_source(reason_code="locked_supportive_biology_yaml_missing")
    if dataset_aliases:
        truth["dataset"] = truth["dataset"].replace(dict(dataset_aliases))
        duplicated = truth.duplicated(["dataset", "observation_id"], keep=False)
        if duplicated.any():
            collisions = sorted(
                set(
                    truth.loc[
                        duplicated, ["dataset", "observation_id"]
                    ].itertuples(index=False, name=None)
                )
            )
            raise ValueError(
                "supportive-biology dataset aliases create duplicate observations: "
                f"{collisions}"
            )
    if table.empty:
        source = truth.copy()
        source["method"] = "No method result"
        source["resource_mode"] = "not_available"
        source["support_status"] = "not_evaluated"
        source["observed_direction"] = ""
        source["evidence_note"] = "biology_support_table_missing"
        source["status"] = "not_estimable"
        source["reason_code"] = "biology_support_table_missing"
        return source
    prepared = pd.DataFrame(
        {
            "dataset": _text(table, ("dataset", "dataset_id"), "unknown"),
            "observation_id": _text(
                table, ("observation_id", "expected_observation_id"), ""
            ),
            "method": _text(table, ("method", "method_id"), "unknown"),
            "resource_mode": _text(table, ("resource_mode",), "unknown"),
            "support_status": _text(
                table, ("support_status", "biology_status", "status"), "not_evaluated"
            ),
            "observed_direction": _text(table, ("observed_direction",), ""),
            "evidence_note": _text(table, ("evidence_note", "note"), ""),
            "source_biology_file": _text(
                table, ("source_biology_file",), ""
            ),
            "rank_scope": _text(
                table, ("rank_scope",), "annotation_only_unbound_rank_scope"
            ),
            "evidence_rank_scope": _text(
                table, ("evidence_rank_scope",), "not_recorded"
            ),
            "n_components_estimable": _numeric(
                table, ("n_components_estimable",)
            ),
            "n_components_strong": _numeric(table, ("n_components_strong",)),
            "n_components_directional": _numeric(
                table, ("n_components_directional",)
            ),
            "n_components_opposite": _numeric(
                table, ("n_components_opposite",)
            ),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    invalid_support = set(prepared["support_status"]).difference(
        BIOLOGY_SUPPORT_STATUSES
    )
    if invalid_support:
        raise ValueError(
            f"biology support contains invalid statuses: {sorted(invalid_support)}"
        )
    known = set(truth[["dataset", "observation_id"]].itertuples(index=False, name=None))
    supplied = set(
        prepared[["dataset", "observation_id"]].itertuples(index=False, name=None)
    )
    unknown = supplied.difference(known)
    if unknown:
        raise ValueError(
            "biology support table contains observations absent from locked YAML: "
            f"{sorted(unknown)}"
        )
    methods = prepared[["method", "resource_mode"]].drop_duplicates()
    grid = (
        truth.assign(_key=1)
        .merge(methods.assign(_key=1), on="_key")
        .drop(columns="_key")
    )
    source = grid.merge(
        prepared,
        on=["dataset", "observation_id", "method", "resource_mode"],
        how="left",
        validate="one_to_one",
    )
    source["support_status"] = source["support_status"].fillna("not_evaluated")
    source["status"] = source["status"].fillna("not_estimable")
    source["reason_code"] = source["reason_code"].fillna("not_evaluated")
    source["observed_direction"] = source["observed_direction"].fillna("")
    source["evidence_note"] = source["evidence_note"].fillna("")
    source["source_biology_file"] = source["source_biology_file"].fillna("")
    return source


def _validate_simulation_records(source: pd.DataFrame) -> None:
    observed = source["status"].isin(OBSERVED_STATUSES)
    values = source.loc[observed, "estimate"]
    if values.isna().any() or np.isinf(values).any():
        raise ValueError("observed simulation metrics require finite estimates")
    classification = observed & source["metric"].isin(
        {
            "auroc",
            "average_precision",
            "differential_auroc",
            "differential_average_precision",
        }
    )
    supplied_counts = source["n_positive"].notna() & source["n_negative"].notna()
    single_class = classification & supplied_counts & (
        source["n_positive"].le(0) | source["n_negative"].le(0)
    )
    if single_class.any():
        raise ValueError("single-class truth cannot have observed AUROC/AUPRC")
    calibrated = observed & source["metric"].isin({"type_i_error", "empirical_fdr"})
    insufficient_null = calibrated & (
        source["n_null_replicates"].isna()
        | source["n_null_replicates"].lt(1000)
    )
    if insufficient_null.any():
        raise ValueError(
            "observed type-I error/FDR requires at least 1000 null replicates"
        )


def _simulation_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="simulation_truth_metrics_missing")
    truth_scope = _text(table, ("truth_scope", "scope"), "")
    invalid = set(truth_scope).difference(ALLOWED_SYNTHETIC_SCOPES)
    if invalid:
        raise ValueError(
            "truth metrics, including AUROC/AUPRC, are allowed only for explicit "
            f"synthetic truth scopes; invalid={sorted(invalid)}"
        )
    dataset = _text(table, ("dataset", "scenario"), "synthetic")
    scenario = _text(table, ("scenario",), "")
    missing_scenario = scenario.eq("")
    scenario.loc[missing_scenario] = dataset.loc[missing_scenario].str.removeprefix(
        "synthetic_"
    )
    identity = pd.DataFrame(
        {
            "dataset": dataset,
            "scenario": scenario,
            "method": _text(table, ("method", "method_id"), "unknown"),
            "resource_mode": _text(table, ("resource_mode",), "not_applicable"),
            "truth_scope": truth_scope,
            "estimand": _text(table, ("estimand",), ""),
            "benchmark_track": _text(table, ("benchmark_track",), ""),
            "record_source": _text(table, ("record_source",), ""),
            "scale_comparability": _text(table, ("scale_comparability",), ""),
            "proxy_semantics": _text(table, ("proxy_semantics",), ""),
            "n_positive": _numeric(table, ("n_positive",)),
            "n_negative": _numeric(table, ("n_negative",)),
            "n_null_replicates": _numeric(table, ("n_null_replicates",)),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    long_mask = pd.Series(False, index=table.index)
    if "metric" in table:
        long_mask = table["metric"].notna() & table["metric"].astype(str).str.strip().ne("")
    frames: list[pd.DataFrame] = []
    if long_mask.any():
        if "estimate" not in table:
            raise ValueError("long-form simulation metrics require an estimate column")
        long = identity.loc[long_mask].copy()
        long["metric"] = table.loc[long_mask, "metric"].astype(str)
        long["estimate"] = pd.to_numeric(
            table.loc[long_mask, "estimate"], errors="coerce"
        )
        frames.append(long)

    wide_mask = ~long_mask
    metric_columns = [
        column
        for column in (
            "average_precision",
            "auroc",
            "top_k_precision",
            "top_k_recall",
            "top_k_f1",
            "top_k_mcc",
            "signed_target_program_macro_auprc",
            "type_i_error",
            "empirical_fdr",
            "coverage_95",
            "direction_power",
        )
        if column in table
        and pd.to_numeric(table.loc[wide_mask, column], errors="coerce").notna().any()
    ]
    for column in metric_columns:
        frame = identity.loc[wide_mask].copy()
        frame["metric"] = column
        frame["estimate"] = pd.to_numeric(
            table.loc[wide_mask, column], errors="coerce"
        )
        nonfinite_observed = frame["status"].isin(OBSERVED_STATUSES) & ~np.isfinite(
            frame["estimate"]
        )
        frame.loc[nonfinite_observed, "status"] = "not_estimable"
        frame.loc[nonfinite_observed, "reason_code"] = f"{column}_not_estimable"
        frames.append(frame)
    if not frames:
        return _ne_source(reason_code="simulation_metric_columns_missing")
    source = pd.concat(frames, ignore_index=True)
    _validate_simulation_records(source)
    return source


def _iteration_source(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _ne_source(reason_code="iteration_comparison_missing")
    source = pd.DataFrame(
        {
            "dataset": _text(table, ("dataset",), "unknown"),
            "method": _text(table, ("method", "method_id"), "CRYCHIC"),
            "metric": _text(table, ("metric",), "unknown"),
            "iteration_from": _text(
                table, ("iteration_from", "baseline_iteration"), "before"
            ),
            "iteration_to": _text(
                table, ("iteration_to", "candidate_iteration"), "after"
            ),
            "before": _numeric(table, ("before", "baseline_value")),
            "after": _numeric(table, ("after", "candidate_value")),
            "metric_direction": _text(
                table, ("metric_direction", "direction"), "higher"
            ),
            "status": _status(table, "not_estimable"),
            "reason_code": _text(table, ("reason_code",), ""),
        }
    )
    invalid_direction = set(source["metric_direction"]).difference({"higher", "lower"})
    if invalid_direction:
        raise ValueError(
            f"iteration metric_direction must be higher/lower: {invalid_direction}"
        )
    observed = source["status"].isin(OBSERVED_STATUSES)
    values = source.loc[observed, ["before", "after"]]
    if values.isna().any().any() or np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("observed iteration rows require finite before/after values")
    sign = source["metric_direction"].map({"higher": 1.0, "lower": -1.0})
    source["signed_improvement"] = math.nan
    source.loc[observed, "signed_improvement"] = (
        source.loc[observed, "after"] - source.loc[observed, "before"]
    ) * sign.loc[observed]
    source["relative_improvement"] = math.nan
    nonzero_baseline = observed & source["before"].ne(0)
    source.loc[nonzero_baseline, "relative_improvement"] = (
        source.loc[nonzero_baseline, "signed_improvement"]
        / source.loc[nonzero_baseline, "before"].abs()
    )
    source["change_class"] = "not_estimable"
    source.loc[observed, "change_class"] = np.select(
        [
            source.loc[observed, "signed_improvement"] > 0,
            source.loc[observed, "signed_improvement"] < 0,
        ],
        ["improved", "regressed"],
        default="unchanged",
    )
    return source


def _panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _empty_panel(axis: mpl.axes.Axes, reason: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(LIGHT_GRAY)
    axis.text(
        0.5,
        0.55,
        "NE",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        color=MID_GRAY,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.35,
        reason,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=MID_GRAY,
        wrap=True,
    )


def _save_figure(figure: Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        figure.savefig(figures_dir / f"{stem}.{suffix}", facecolor="white")
    figure.savefig(figures_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)


def _save_source(data: pd.DataFrame, source_dir: Path, stem: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(source_dir / f"{stem}.csv", index=False, na_rep="")


def _plot_design(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    data = source.loc[~source["method"].eq("NE")].copy() if "method" in source else source.copy()
    data["dataset_label"] = _short_dataset_labels(data["dataset"])
    data["scope_order"] = data["truth_scope"].map({"real_data": 0, "simulation": 1}).fillna(2)
    data = data.sort_values(["scope_order", "dataset"], kind="stable").reset_index(drop=True)
    y = np.arange(len(data))

    if data.empty or data["n_cells"].isna().all():
        _empty_panel(axes[0, 0], "cell counts unavailable")
    else:
        axes[0, 0].barh(y, data["n_cells"], color=BLUE)
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yticks(y, data["dataset_label"], fontsize=7)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel("Cells/nuclei (log scale)")
    axes[0, 0].set_title("Dataset scale")
    _panel_label(axes[0, 0], "A")

    if data.empty or data[["n_samples", "n_subjects"]].isna().all().all():
        _empty_panel(axes[0, 1], "sample and subject support unavailable")
    else:
        axes[0, 1].barh(y - 0.18, data["n_samples"], height=0.34, color=SKY, label="Samples")
        axes[0, 1].barh(y + 0.18, data["n_subjects"], height=0.34, color=ORANGE, label="Subjects")
        axes[0, 1].set_yticks(y, data["dataset_label"], fontsize=7)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel("Count")
        axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_title("Biological-unit support")
    _panel_label(axes[0, 1], "B")

    if data.empty or data[["n_contexts", "n_cell_types"]].isna().all().all():
        _empty_panel(axes[1, 0], "context and cell-type counts unavailable")
    else:
        axes[1, 0].barh(y - 0.18, data["n_contexts"], height=0.34, color=GREEN, label="Contexts")
        axes[1, 0].barh(y + 0.18, data["n_cell_types"], height=0.34, color=PINK, label="Cell types")
        axes[1, 0].set_yticks(y, data["dataset_label"], fontsize=7)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel("Count")
        axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("Design dimensions")
    _panel_label(axes[1, 0], "C")

    real = data.loc[data["truth_scope"].eq("real_data")].copy()
    if real.empty:
        _empty_panel(axes[1, 1], str(source.iloc[0].get("reason_code", "no data")))
    else:
        real_y = np.arange(len(real))
        colors = [GREEN if status in OBSERVED_STATUSES else LIGHT_GRAY for status in real["status"]]
        axes[1, 1].barh(real_y, np.ones(len(real)), color=colors)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks(real_y, real["dataset_label"], fontsize=7)
        axes[1, 1].invert_yaxis()
        reason_labels = {
            "mixed_paired_unpaired_multi_region_design_unsupported": "NE: mixed paired/unpaired multi-region design",
            "pooled_nonindependent_objects_no_subject_id": "NE: pooled objects; no independent subject ID",
        }
        for row, (_, item) in enumerate(real.iterrows()):
            if item["status"] in OBSERVED_STATUSES:
                label = str(item["design_type"])
            else:
                label = reason_labels.get(
                    str(item["reason_code"]), f"NE: {item['reason_code']}"
                )
            axes[1, 1].text(0.02, row, label, va="center", fontsize=7)
    axes[1, 1].set_title("Real-data design audit (NE retained)")
    _panel_label(axes[1, 1], "D")
    figure.suptitle("Dataset design, biological replication, and estimability")
    figure.subplots_adjust(
        left=0.17, right=0.98, top=0.92, bottom=0.08, hspace=0.38, wspace=0.34
    )
    _save_figure(figure, figures_dir, "figure01_design_estimability")


def _plot_coverage(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
    observed = source.loc[~source["method"].eq("NE")].copy()
    dataset_code = _short_dataset_labels(observed["dataset"]).replace(
        {"MS CA vs Ctrl": "MS"}
    )
    method_code = _display_method_names(observed["method"]).replace(
        {"NicheNet-prior": "NN proxy", "No adapter run": "No run"}
    )
    arm_code = observed["resource_mode"].replace({"H-common": "H", "native": "N"})
    arm_code.loc[observed["analysis_track"].eq("ligand_target_program")] = "P"
    observed["compact_label"] = (
        dataset_code + " | " + method_code + " [" + arm_code + "]"
    )
    for axis, column, title, label in (
        (axes[0, 0], "resource_coverage_fraction", "Frozen-resource coverage", "A"),
        (
            axes[0, 1],
            "comparison_coverage_fraction",
            "Comparison-eligible score coverage",
            "B",
        ),
    ):
        data = observed.dropna(subset=[column])
        if data.empty:
            _empty_panel(axis, f"{column} unavailable")
        else:
            axis.barh(
                data["compact_label"],
                data[column],
                color=BLUE if label == "A" else GREEN,
            )
            axis.axvline(1.0, color=BLACK, linewidth=0.8)
            axis.set_xlim(0, 1.05)
            axis.set_xlabel("Fraction")
            axis.tick_params(axis="y", labelsize=6)
        axis.set_title(title)
        _panel_label(axis, label)

    status_columns = [
        "observed_rows",
        "not_predicted_rows",
        "missing_rows",
        "cell_type_missing_rows",
        "not_estimable_rows",
        "failed_rows",
        "not_supported_rows",
    ]
    status_data = observed.loc[:, ["compact_label", *status_columns]].copy()
    status_data["label"] = status_data["compact_label"]
    status_data = status_data.groupby("label", sort=False)[status_columns].sum()
    totals = status_data.sum(axis=1).replace(0, np.nan)
    if status_data.empty or totals.isna().all():
        _empty_panel(axes[1, 0], "explicit row-status counts unavailable")
    else:
        fractions = status_data.loc[totals.notna()].div(totals.dropna(), axis=0)
        fractions.plot.barh(
            stacked=True,
            ax=axes[1, 0],
            color=[GREEN, SKY, LIGHT_GRAY, ORANGE, MID_GRAY, VERMILLION, PINK],
        )
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_xlabel("Fraction of rows")
        axes[1, 0].set_ylabel("")
        axes[1, 0].tick_params(axis="y", labelsize=5.5)
        axes[1, 0].legend(
            ncol=3,
            fontsize=5.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
        )
    axes[1, 0].set_title("Explicit output states")
    _panel_label(axes[1, 0], "C")

    if observed.empty:
        _empty_panel(
            axes[1, 1], str(source.iloc[0].get("reason_code", "no coverage data"))
        )
    else:
        rows = observed.assign(key=observed["compact_label"])
        status_colors = {
            "observed": GREEN,
            "complete": GREEN,
            "not_estimable": MID_GRAY,
            "not_supported": PINK,
            "failed": VERMILLION,
        }
        colors = [status_colors.get(value, ORANGE) for value in rows["status"]]
        axes[1, 1].barh(rows["key"], np.ones(len(rows)), color=colors)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].tick_params(axis="y", labelsize=5.5)
        reason_labels = {
            "dataset_has_no_adapter_runs": "NE: no adapter run",
            "track_b_ligand_target_program_not_lr_stlr_comparable": (
                "NE: proxy diagnostic is not Track A LR"
            ),
        }
        for index, (_, row) in enumerate(rows.iterrows()):
            text = str(row["status"])
            if row["status"] not in OBSERVED_STATUSES and row["reason_code"]:
                text = reason_labels.get(
                    str(row["reason_code"]), f"{text}: {row['reason_code']}"
                )
            axes[1, 1].text(0.02, index, text, va="center", fontsize=5.5)
    axes[1, 1].set_title("Method/resource run state")
    _panel_label(axes[1, 1], "D")
    figure.suptitle(
        "Resource coverage and explicit method failure states "
        "(H=H-common; N=native LR; P=proxy)"
    )
    figure.subplots_adjust(
        left=0.20, right=0.98, top=0.92, bottom=0.12, hspace=0.44, wspace=0.36
    )
    _save_figure(figure, figures_dir, "figure02_coverage_status")


def _plot_loso(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(12.2, 6.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    observed = source.loc[source["status"].isin(OBSERVED_STATUSES)].copy()
    if observed.empty:
        _empty_panel(axes[0], str(source.iloc[0].get("reason_code", "no LOSO result")))
    else:
        observed["dataset_label"] = _short_dataset_labels(observed["dataset"])
        observed["label"] = (
            observed["dataset_label"]
            + " | "
            + observed["method"]
            + " | "
            + observed["resource_mode"]
        )
        observed = observed.sort_values("estimate", kind="stable")
        positions = np.arange(len(observed))
        errors = np.vstack(
            [
                observed["estimate"] - observed["ci_lower"],
                observed["ci_upper"] - observed["estimate"],
            ]
        )
        axes[0].errorbar(
            observed["estimate"],
            positions,
            xerr=errors,
            fmt="o",
            color=BLUE,
            ecolor=MID_GRAY,
            capsize=3,
        )
        axes[0].axvline(0, color=BLACK, linewidth=0.8, linestyle="--")
        axes[0].set_yticks(positions, observed["label"])
        axes[0].set_xlim(-1.05, 1.05)
        axes[0].set_xlabel(
            "Differential-rank stability (design-specific 95% interval)"
        )
    axes[0].set_title("Primary real-data endpoint")
    _panel_label(axes[0], "A")

    ne = source.loc[~source["status"].isin(OBSERVED_STATUSES)]
    axes[1].axis("off")
    axes[1].set_title("Coverage and NE reasons")
    if ne.empty and not observed.empty:
        lines = [
            f"Observed systems: {len(observed)}",
            f"Datasets: {observed['dataset'].nunique()}",
            "Paired: subject LOSO; independent groups: stratified split-half",
            "Aggregation: equal families, then biological subjects",
            "Consensus is not treated as truth.",
        ]
    else:
        lines = ["NE rows are retained:"]
        for _, row in ne.head(12).iterrows():
            lines.append(
                f"{row['dataset']} | {row['method']}: "
                f"{row['reason_code'] or row['status']}"
            )
    axes[1].text(0, 0.95, "\n".join(lines), va="top", fontsize=8, wrap=True)
    _panel_label(axes[1], "B")
    figure.suptitle("Subject-level differential-rank stability")
    figure.subplots_adjust(left=0.27, right=0.96, top=0.88, bottom=0.14, wspace=0.38)
    _save_figure(figure, figures_dir, "figure03_loso_primary")


def _plot_robustness(
    stability: pd.DataFrame,
    concordance: pd.DataFrame,
    performance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.0))
    stable = stability.loc[
        ~stability["method"].eq("NE")
        & stability["truth_scope"].eq("real_data")
        & stability["analysis_track"].eq("lr_stlr")
        & stability["resource_mode"].eq("H-common")
        & stability["status"].isin(OBSERVED_STATUSES)
        & stability["metric"].isin({"median_spearman", "median_top_k_jaccard"})
    ].dropna(subset=["estimate"])
    if stable.empty:
        _empty_panel(axes[0, 0], str(stability.iloc[0].get("reason_code", "NE")))
    else:
        stable = stable.copy()
        method_names = stable["method"].replace(
            {
                "cellchat": "CellChat",
                "cellphonedb": "CellPhoneDB",
                "liana_rank_aggregate": "LIANA",
                "crychic": "CRYCHIC",
            }
        )
        context = stable["context"].replace("", "context")
        stable["label"] = (
            _short_dataset_labels(stable["dataset"])
            + " | "
            + method_names
            + " | "
            + context
        )
        metric_order = ["median_spearman", "median_top_k_jaccard"]
        matrix = stable.pivot_table(
            index="label",
            columns="metric",
            values="estimate",
            aggfunc="first",
            sort=False,
        ).reindex(columns=metric_order)
        matrix_values = matrix.to_numpy(dtype=float)
        image = axes[0, 0].imshow(
            matrix_values,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        axes[0, 0].set_xticks(
            np.arange(2), ["Rank Spearman", "Top-k Jaccard"]
        )
        axes[0, 0].set_yticks(np.arange(len(matrix)), matrix.index, fontsize=6)
        for row in range(len(matrix)):
            for column in range(2):
                value = float(matrix_values[row, column])
                label = "NE" if not math.isfinite(value) else f"{value:.2f}"
                axes[0, 0].text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if math.isfinite(value) and value > 0.58 else BLACK,
                )
        figure.colorbar(image, ax=axes[0, 0], fraction=0.035, pad=0.02)
    axes[0, 0].set_title("Real-data within-context reproducibility")
    _panel_label(axes[0, 0], "A")

    concordant = concordance.loc[
        ~concordance["method_left"].eq("NE")
        & concordance["truth_scope"].eq("real_data")
        & concordance["analysis_track"].eq("lr_stlr")
        & concordance["resource_mode"].eq("H-common")
        & concordance["status"].isin(OBSERVED_STATUSES)
        & concordance["effect_spearman"].notna()
    ].copy()
    if concordant.empty:
        _empty_panel(axes[0, 1], str(concordance.iloc[0].get("reason_code", "NE")))
    else:
        display = {
            "cellchat": "CellChat",
            "cellphonedb": "CellPhoneDB",
            "liana_rank_aggregate": "LIANA",
            "crychic": "CRYCHIC",
        }
        concordant["label"] = (
            _short_dataset_labels(concordant["dataset"])
            + " | "
            + concordant["method_left"].replace(display)
            + " vs "
            + concordant["method_right"].replace(display)
        )
        concordant = concordant.sort_values(
            ["dataset", "effect_spearman"], kind="stable"
        )
        colors = [BLUE if value >= 0 else VERMILLION for value in concordant["effect_spearman"]]
        axes[0, 1].barh(
            concordant["label"], concordant["effect_spearman"], color=colors
        )
        axes[0, 1].axvline(0, color=BLACK, linewidth=0.8)
        axes[0, 1].set_xlim(-1.02, 1.02)
        axes[0, 1].set_xlabel("Effect-rank Spearman (descriptive, not accuracy)")
        axes[0, 1].tick_params(axis="y", labelsize=6)
    axes[0, 1].set_title("Real-data cross-method concordance by dataset")
    _panel_label(axes[0, 1], "B")

    perf = performance.loc[
        ~performance["method"].eq("NE")
        & performance["truth_scope"].eq("real_data")
    ].copy()
    method_names = perf["method"].replace(
        {
            "cellchat": "CellChat",
            "cellphonedb": "CellPhoneDB",
            "liana_rank_aggregate": "LIANA",
            "crychic": "CRYCHIC",
            "nichenet_prior_activity": "NicheNet-prior proxy",
        }
    )
    track_labels = perf["analysis_track"].replace(
        {"lr_stlr": "Track A", "ligand_target_program": "proxy diagnostic"}
    )
    runtime_roles = perf["method_runtime_role"].replace(
        {
            "method_total": "method total",
            "core_fit": "core fit",
            "source_pipeline_total": "source pipeline total",
            "not_available": "runtime NE",
        }
    )
    perf["label"] = (
        method_names
        + " | "
        + perf["resource_mode"]
        + " | "
        + track_labels
        + " | "
        + runtime_roles
    )
    perf["dataset_label"] = _short_dataset_labels(perf["dataset"])
    categories = list(dict.fromkeys(perf["label"]))
    category_positions = {label: index for index, label in enumerate(categories)}
    dataset_order = list(dict.fromkeys(perf["dataset_label"]))
    offsets = np.linspace(-0.16, 0.16, max(len(dataset_order), 1))
    dataset_offsets = dict(zip(dataset_order, offsets, strict=True))
    dataset_colors = {
        dataset: PALETTE[index % len(PALETTE)]
        for index, dataset in enumerate(dataset_order)
    }
    runtime = perf.dropna(subset=["median_wall_time_seconds"])
    if runtime.empty:
        _empty_panel(axes[1, 0], str(performance.iloc[0].get("reason_code", "NE")))
    else:
        for dataset, group in runtime.groupby("dataset_label", sort=False):
            y = [
                category_positions[label] + dataset_offsets[dataset]
                for label in group["label"]
            ]
            axes[1, 0].scatter(
                group["median_wall_time_seconds"],
                y,
                label=dataset,
                color=dataset_colors[dataset],
                s=28,
            )
        readback = perf.dropna(
            subset=["median_adapter_readback_wall_time_seconds"]
        )
        if not readback.empty:
            y = [
                category_positions[label] + dataset_offsets[dataset]
                for label, dataset in readback[["label", "dataset_label"]].itertuples(
                    index=False, name=None
                )
            ]
            axes[1, 0].scatter(
                readback["median_adapter_readback_wall_time_seconds"],
                y,
                marker="x",
                color=BLACK,
                s=34,
                label="adapter readback only",
            )
        axes[1, 0].set_yticks(np.arange(len(categories)), categories, fontsize=6)
        if (runtime["median_wall_time_seconds"] > 0).all():
            axes[1, 0].set_xscale("log")
        axes[1, 0].set_xlabel("Wall time (s; log scale; descriptive only)")
        axes[1, 0].legend(title="Dataset", fontsize=6, title_fontsize=6)
    axes[1, 0].set_title("Dataset-specific runtime at recorded thread caps")
    _panel_label(axes[1, 0], "C")

    memory = perf.dropna(subset=["median_peak_rss_mb"])
    if memory.empty:
        _empty_panel(axes[1, 1], "peak RSS unavailable")
    else:
        for dataset, group in memory.groupby("dataset_label", sort=False):
            y = [
                category_positions[label] + dataset_offsets[dataset]
                for label in group["label"]
            ]
            axes[1, 1].scatter(
                group["median_peak_rss_mb"],
                y,
                color=dataset_colors[dataset],
                s=28,
            )
        axes[1, 1].set_yticks(np.arange(len(categories)), categories, fontsize=6)
        if (memory["median_peak_rss_mb"] > 0).all():
            axes[1, 1].set_xscale("log")
        axes[1, 1].set_xlabel("Peak RSS (MB; log scale)")
        axes[1, 1].text(
            0.02,
            0.02,
            f"Measured {len(memory)}/{len(perf)} real-data runs; NA not imputed",
            transform=axes[1, 1].transAxes,
            fontsize=6,
            color=MID_GRAY,
        )
    axes[1, 1].set_title("Dataset-specific peak RSS (missing retained)")
    _panel_label(axes[1, 1], "D")
    figure.suptitle("Robustness, method concordance, and computational cost")
    figure.subplots_adjust(
        left=0.28, right=0.97, top=0.92, bottom=0.10, hspace=0.46, wspace=0.48
    )
    _save_figure(figure, figures_dir, "figure04_robustness_performance")


def _plot_biology(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.2, max(4.8, min(11.0, len(source) * 0.24))))
    if "observation_id" not in source or source["method"].eq("NE").all():
        _empty_panel(axis, str(source.iloc[0].get("reason_code", "biology NE")))
    else:
        source = source.copy()
        source["method_label"] = source["method"] + " | " + source["resource_mode"]
        source["observation_label"] = (
            source["dataset"] + " | " + source["observation_id"]
        )
        methods = list(dict.fromkeys(source["method_label"]))
        observations = list(dict.fromkeys(source["observation_label"]))
        categories = {
            "discordant": 0,
            "not_estimable": 1,
            "not_covered": 2,
            "not_evaluated": 3,
            "partial": 4,
            "supported": 5,
        }
        symbols = {
            "discordant": "-",
            "not_estimable": "NE",
            "not_covered": "NC",
            "not_evaluated": "NA",
            "partial": "~",
            "supported": "+",
        }
        matrix = np.full((len(observations), len(methods)), categories["not_evaluated"])
        labels: dict[tuple[int, int], str] = {}
        for _, row in source.iterrows():
            y = observations.index(row["observation_label"])
            x = methods.index(row["method_label"])
            category = str(row["support_status"])
            if category not in categories:
                category = "not_evaluated"
            matrix[y, x] = categories[category]
            opposite = row.get("n_components_opposite", math.nan)
            if pd.notna(opposite) and float(opposite) > 0:
                labels[(y, x)] = f"{symbols[category]}\n{int(opposite)}opp"
            else:
                labels[(y, x)] = symbols[category]
        cmap = ListedColormap(
            [VERMILLION, MID_GRAY, SKY, VERY_LIGHT_GRAY, YELLOW, GREEN]
        )
        axis.imshow(matrix, cmap=cmap, vmin=0, vmax=5, aspect="auto")
        axis.set_xticks(np.arange(len(methods)), methods, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(observations)), observations)
        for (y, x), label in labels.items():
            axis.text(x, y, label, ha="center", va="center", fontsize=7)
        axis.set_xlabel(
            "Method and resource arm  |  + supported; ~ partial; - discordant; "
            "NC not covered; NE not estimable; NA not evaluated"
        )
    axis.set_title("Preregistered supportive biology (not edge-level ground truth)")
    figure.subplots_adjust(left=0.35, right=0.97, top=0.91, bottom=0.22)
    _save_figure(figure, figures_dir, "figure05_supportive_biology")


def _plot_simulation_generic(
    source: pd.DataFrame, axes: Sequence[mpl.axes.Axes]
) -> None:
    observed = source.loc[~source["method"].eq("NE")].dropna(subset=["estimate"])
    if observed.empty:
        reason = str(source.iloc[0].get("reason_code", "simulation truth unavailable"))
        _empty_panel(axes[0], reason)
        _empty_panel(axes[1], reason)
    else:
        rank_metrics = observed.loc[
            observed["metric"].str.contains(
                "auroc|average_precision|auprc|precision|recall|f1|mcc",
                case=False,
                regex=True,
            )
        ]
        calibration = observed.loc[~observed.index.isin(rank_metrics.index)]
        for axis, data, title, label in (
            (axes[0], rank_metrics, "Known-truth recovery", "A"),
            (axes[1], calibration, "Calibration and direction", "B"),
        ):
            if data.empty:
                _empty_panel(axis, "metric class not available")
            else:
                pivot = data.pivot_table(
                    index=["method", "resource_mode"],
                    columns="metric",
                    values="estimate",
                    aggfunc="mean",
                )
                pivot.plot.bar(ax=axis, color=list(PALETTE[: len(pivot.columns)]))
                axis.set_xlabel("")
                axis.set_ylabel("Metric value")
                axis.tick_params(axis="x", rotation=30)
                axis.legend(fontsize=6)
            axis.set_title(title)
            _panel_label(axis, label)


def _plot_simulation_tracks(
    source: pd.DataFrame, axes: Sequence[mpl.axes.Axes]
) -> bool:
    track_a = source.loc[
        source["record_source"].eq("track_a_differential_truth")
        & source["scenario"].eq("active")
        & source["status"].isin(OBSERVED_STATUSES)
        & source["metric"].isin(
            {
                "differential_auroc",
                "differential_average_precision",
                "estimable_edge_coverage_fraction",
            }
        )
    ].copy()
    track_b = source.loc[
        source["record_source"].eq("track_b_receiver_program_truth")
        & source["status"].isin(OBSERVED_STATUSES)
        & source["metric"].isin(
            {
                "cxcl10_receiver_program_effect",
                "cxcl10_receiver_program_percentile",
            }
        )
    ].copy()
    if track_a.empty or track_b.empty:
        return False

    method_order = [
        method
        for method in (
            "cellchat",
            "cellphonedb",
            "crychic",
            "liana_rank_aggregate",
        )
        if method in set(track_a["method"])
    ]
    estimand_order = [
        estimand
        for estimand in (
            "paired_oriented_native_score_difference",
            "paired_rank_strength_difference",
        )
        if estimand in set(track_a["estimand"])
    ]
    metric_order = [
        "differential_auroc",
        "differential_average_precision",
        "estimable_edge_coverage_fraction",
    ]
    if not method_order or not estimand_order:
        return False

    columns = [(estimand, metric) for estimand in estimand_order for metric in metric_order]
    matrix = np.full((len(method_order), len(columns)), np.nan)
    for row_index, method in enumerate(method_order):
        for column_index, (estimand, metric) in enumerate(columns):
            values = track_a.loc[
                track_a["method"].eq(method)
                & track_a["estimand"].eq(estimand)
                & track_a["metric"].eq(metric),
                "estimate",
            ].dropna()
            if len(values) == 1:
                matrix[row_index, column_index] = float(values.iloc[0])

    axis = axes[0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "crychic_recovery", [VERY_LIGHT_GRAY, SKY, BLUE]
    )
    image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    metric_labels = ("AUROC", "Avg. precision", "Coverage")
    column_labels = [
        f"{'Native' if estimand == estimand_order[0] else 'Rank'}\n{metric_labels[index % 3]}"
        for index, (estimand, _) in enumerate(columns)
    ]
    display_methods = {
        "cellchat": "CellChat",
        "cellphonedb": "CellPhoneDB",
        "crychic": "CRYCHIC",
        "liana_rank_aggregate": "LIANA",
    }
    axis.set_xticks(np.arange(len(columns)), column_labels)
    axis.set_yticks(
        np.arange(len(method_order)),
        [display_methods.get(method, method) for method in method_order],
    )
    axis.tick_params(axis="x", rotation=28)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "NE" if not math.isfinite(value) else f"{value:.2f}"
            color = "white" if math.isfinite(value) and value >= 0.58 else BLACK
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    axis.axvline(2.5, color="white", linewidth=2.2)
    axis.set_title("Track A: active LR diagnostic (1 positive / 44 negatives)")
    axis.set_xlabel("Native scale is primary; rank strength is a sensitivity estimand")
    _panel_label(axis, "A")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("Metric value", fontsize=7)

    effect = track_b.loc[
        track_b["metric"].eq("cxcl10_receiver_program_effect")
    ].drop_duplicates("scenario")
    percentile = (
        track_b.loc[track_b["metric"].eq("cxcl10_receiver_program_percentile")]
        .drop_duplicates("scenario")
        .set_index("scenario")["estimate"]
    )
    scenario_order = [
        scenario
        for scenario in (
            "active",
            "global_null",
            "abundance_only",
            "ligand_only",
            "target_only",
            "receptor_knockout",
            "receiver_autonomous",
        )
        if scenario in set(effect["scenario"])
    ]
    effect = effect.set_index("scenario").reindex(scenario_order)
    effect_values = effect["estimate"].to_numpy(dtype=float)
    colors = [
        BLUE
        if scenario == "active"
        else ORANGE
        if scenario in {"target_only", "receptor_knockout"}
        else GREEN
        if value > 0
        else VERMILLION
        for scenario, value in zip(scenario_order, effect_values, strict=True)
    ]
    axis = axes[1]
    y = np.arange(len(scenario_order))
    axis.barh(y, effect_values, color=colors)
    axis.axvline(0.0, color=BLACK, linewidth=0.8)
    labels = {
        "active": "Active",
        "global_null": "Global null",
        "abundance_only": "Abundance only",
        "ligand_only": "Ligand only",
        "target_only": "Target only*",
        "receptor_knockout": "Receptor KO*",
        "receiver_autonomous": "Receiver autonomous",
    }
    axis.set_yticks(y, [labels.get(scenario, scenario) for scenario in scenario_order])
    axis.invert_yaxis()
    span = max(float(np.nanmax(np.abs(effect_values))), 0.01)
    axis.set_xlim(-span * 1.35, span * 1.55)
    for index, (scenario, value) in enumerate(
        zip(scenario_order, effect_values, strict=True)
    ):
        pct = percentile.get(scenario, math.nan)
        pct_label = f"; pct={pct:.2f}" if math.isfinite(float(pct)) else ""
        text_x = value + 0.025 * span if value >= 0 else 0.025 * span
        axis.text(
            text_x,
            index,
            f"{value:.3f}{pct_label}",
            ha="left",
            va="center",
            fontsize=7,
        )
    axis.set_title("Track B proxy diagnostic: CXCL10 receiver program")
    axis.set_xlabel(
        "Source-agnostic frozen-prior effect\n"
        "not the preregistered Track B primary; * LR-edge-negative"
    )
    _panel_label(axis, "B")
    return True


def _plot_simulation(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.8),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    if not _plot_simulation_tracks(source, axes):
        _plot_simulation_generic(source, axes)
    figure.suptitle("Synthetic and perturbation endpoints with explicit truth")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.38)
    _save_figure(figure, figures_dir, "figure06_simulation_truth")


def _plot_iteration(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 7.0))
    observed = source.loc[
        ~source["method"].eq("NE")
        & source["status"].isin(OBSERVED_STATUSES)
        & source["before"].notna()
        & source["after"].notna()
    ].copy()
    ne = source.loc[
        ~source["method"].eq("NE") & ~source["status"].isin(OBSERVED_STATUSES)
    ].copy()
    if observed.empty:
        reason = str(
            source.iloc[0].get("reason_code", "iteration comparison unavailable")
        )
        _empty_panel(axes[0], reason)
        _empty_panel(axes[1], reason)
    else:
        observed["label"] = (
            _short_dataset_labels(observed["dataset"])
            + " | "
            + observed["metric"]
        )
        observed["after_over_before"] = observed["after"] / observed["before"]
        y = np.arange(len(observed))
        for index, (_, row) in enumerate(observed.iterrows()):
            color = (
                GREEN
                if row["change_class"] == "improved"
                else VERMILLION
                if row["change_class"] == "regressed"
                else MID_GRAY
            )
            ratio = float(row["after_over_before"])
            axes[0].plot([1.0, ratio], [index, index], color=LIGHT_GRAY)
            axes[0].scatter(1.0, index, color=MID_GRAY, marker="o")
            axes[0].scatter(ratio, index, color=color, marker="D")
        axes[0].set_yticks(y, observed["label"])
        axes[0].axvline(1.0, color=BLACK, linewidth=0.8, linestyle="--")
        axes[0].set_xlabel(
            "After / before (raw ratio; color encodes desired direction)"
        )
        axes[0].set_title("Observed comparisons normalized to baseline")
        _panel_label(axes[0], "A")

        ne["label"] = (
            _short_dataset_labels(ne["dataset"]) + " | " + ne["metric"]
        )
        rows = pd.concat([observed, ne], ignore_index=True, sort=False)
        positions = np.arange(len(rows))
        observed_positions = positions[: len(observed)]
        colors = [
            GREEN
            if value > 0
            else VERMILLION
            if value < 0
            else MID_GRAY
            for value in observed["relative_improvement"]
        ]
        axes[1].barh(
            observed_positions,
            observed["relative_improvement"] * 100,
            color=colors,
        )
        axes[1].axvline(0, color=BLACK, linewidth=0.8)
        axes[1].set_yticks(positions, rows["label"], fontsize=6)
        axes[1].set_xlabel("Direction-aware relative change (%)")
        axes[1].set_title("Observed changes; NE rows retain reasons")
        if not ne.empty:
            observed_span = observed["relative_improvement"].abs().max() * 100
            text_x = max(float(observed_span) * 0.06, 1.0)
            for position, (_, row) in zip(
                positions[len(observed) :], ne.iterrows(), strict=True
            ):
                axes[1].text(
                    text_x,
                    position,
                    f"NE: {row['reason_code'] or row['status']}",
                    va="center",
                    fontsize=5.5,
                    color=MID_GRAY,
                )
        _panel_label(axes[1], "B")
    figure.suptitle("Iteration audit: controlled output changes and explicit NE states")
    figure.subplots_adjust(
        left=0.26, right=0.98, top=0.89, bottom=0.12, wspace=0.48
    )
    _save_figure(figure, figures_dir, "figure07_iteration_comparison")


def _plot_rank_stability(
    agreement: pd.DataFrame,
    curve: pd.DataFrame,
    intervals: pd.DataFrame,
    tiers: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.2))
    display_methods = {
        "cellchat": "CellChat",
        "cellphonedb": "CellPhoneDB",
        "liana_rank_aggregate": "LIANA",
        "crychic": "CRYCHIC",
    }

    observed_agreement = agreement.loc[
        agreement["status"].isin(OBSERVED_STATUSES)
        & agreement["analysis_track"].eq("lr_stlr")
        & agreement["estimate"].notna()
    ].copy()
    if observed_agreement.empty:
        _empty_panel(
            axes[0, 0], str(agreement.iloc[0].get("reason_code", "NE"))
        )
    else:
        observed_agreement["label"] = (
            _short_dataset_labels(observed_agreement["dataset"])
            + " | "
            + observed_agreement["method"].replace(display_methods)
            + " | "
            + observed_agreement["resource_mode"]
            + " | "
            + observed_agreement["rank_scope"]
            + " | "
            + observed_agreement["ranking_level"]
            + " | "
            + observed_agreement["metric"].replace(
                {
                    "rank_biased_overlap": "RBO",
                    "weighted_kendall_tau": "weighted tau",
                }
            )
        )
        observed_agreement = observed_agreement.sort_values(
            [
                "dataset",
                "method",
                "resource_mode",
                "rank_scope",
                "ranking_level",
                "metric",
            ],
            kind="stable",
        )
        positions = np.arange(len(observed_agreement))
        errors = np.vstack(
            [
                observed_agreement["estimate"]
                - observed_agreement["envelope_lower"],
                observed_agreement["envelope_upper"]
                - observed_agreement["estimate"],
            ]
        )
        axes[0, 0].errorbar(
            observed_agreement["estimate"],
            positions,
            xerr=errors,
            fmt="o",
            color=BLUE,
            ecolor=MID_GRAY,
            capsize=2,
            markersize=4,
        )
        axes[0, 0].axvline(0, color=BLACK, linewidth=0.7)
        axes[0, 0].set_yticks(
            positions, observed_agreement["label"], fontsize=5.5
        )
        axes[0, 0].set_xlim(-1.05, 1.05)
        axes[0, 0].set_xlabel("Median split-half agreement (95% repeat envelope)")
    axes[0, 0].set_title("Top-sensitive rank agreement")
    _panel_label(axes[0, 0], "A")

    observed_curve = curve.loc[
        curve["status"].isin(OBSERVED_STATUSES)
        & curve["analysis_track"].eq("lr_stlr")
        & curve["ranking_level"].isin(
            {"lr_family", "lr", "sender_receiver_pair"}
        )
        & curve["estimate"].notna()
    ].copy()
    if observed_curve.empty:
        _empty_panel(axes[0, 1], str(curve.iloc[0].get("reason_code", "NE")))
    else:
        group_columns = [
            "dataset",
            "method",
            "resource_mode",
            "rank_scope",
            "ranking_level",
        ]
        for index, (keys, group) in enumerate(
            observed_curve.groupby(group_columns, sort=False, observed=True)
        ):
            dataset, method, resource_mode, rank_scope, level = keys
            label = (
                f"{_short_dataset_labels(pd.Series([dataset])).iloc[0]} | "
                f"{display_methods.get(str(method), method)} | {resource_mode} | "
                f"{rank_scope} | {level}"
            )
            group = group.sort_values("k", kind="stable")
            axes[0, 1].plot(
                group["k"],
                group["estimate"],
                label=label,
                color=PALETTE[index % len(PALETTE)],
                linewidth=1.2,
            )
        axes[0, 1].set_xlim(left=1)
        axes[0, 1].set_ylim(-0.02, 1.02)
        axes[0, 1].set_xlabel(
            "Top-k cutoff (LR family/pair 1-25; LR 1-100)"
        )
        axes[0, 1].set_ylabel("Median split-half Jaccard")
        axes[0, 1].legend(fontsize=5, loc="best")
    axes[0, 1].set_title("Complete top-k stability curves")
    _panel_label(axes[0, 1], "B")

    observed_tiers = tiers.loc[
        tiers["analysis_track"].eq("lr_stlr")
        & tiers["tier"].isin(
            {"stable_top_k", "possible_top_k", "unstable", "stable_below_top_k"}
        )
    ].copy()
    if observed_tiers.empty:
        _empty_panel(axes[1, 0], str(tiers.iloc[0].get("reason_code", "NE")))
    else:
        observed_tiers["label"] = (
            _short_dataset_labels(observed_tiers["dataset"])
            + " | "
            + observed_tiers["method"].replace(display_methods)
            + " | "
            + observed_tiers["resource_mode"]
            + " | "
            + observed_tiers["rank_scope"]
            + " | "
            + observed_tiers["ranking_level"]
        )
        tier_order = [
            "stable_top_k",
            "possible_top_k",
            "unstable",
            "stable_below_top_k",
        ]
        counts = (
            observed_tiers.groupby(["label", "tier"], sort=False, observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=tier_order, fill_value=0)
        )
        left: np.ndarray = np.zeros(len(counts), dtype=float)
        tier_colors = [GREEN, ORANGE, VERMILLION, LIGHT_GRAY]
        for tier, color in zip(tier_order, tier_colors, strict=True):
            axes[1, 0].barh(
                counts.index,
                counts[tier],
                left=left,
                color=color,
                label=tier.replace("_", " "),
            )
            left += counts[tier].to_numpy(dtype=float)
        axes[1, 0].set_xlabel("Number of frozen-universe items")
        axes[1, 0].tick_params(axis="y", labelsize=5.5)
        axes[1, 0].legend(fontsize=5, ncol=2, loc="best")
    axes[1, 0].set_title("Bootstrap-supported stable tiers")
    _panel_label(axes[1, 0], "C")

    observed_intervals = intervals.loc[
        intervals["status"].isin(OBSERVED_STATUSES)
        & intervals["analysis_track"].eq("lr_stlr")
        & intervals["median_rank"].notna()
        & intervals["top_k_frequency"].notna()
    ].copy()
    if observed_intervals.empty:
        _empty_panel(
            axes[1, 1], str(intervals.iloc[0].get("reason_code", "NE"))
        )
    else:
        observed_intervals = (
            observed_intervals.sort_values(
                ["top_k_frequency", "median_rank"],
                ascending=[False, True],
                kind="stable",
            )
            .groupby(
                [
                    "dataset",
                    "method",
                    "resource_mode",
                    "rank_scope",
                    "ranking_level",
                ],
                # Keep each resource arm and ranking estimand visually distinct.
                sort=False,
                observed=True,
            )
            .head(10)
        )
        for keys, group in observed_intervals.groupby(
            ["dataset", "method", "resource_mode", "rank_scope", "ranking_level"],
            sort=False,
            observed=True,
        ):
            dataset, method, resource_mode, rank_scope, level = keys
            xerr = np.vstack(
                [
                    group["median_rank"] - group["lower_rank"],
                    group["upper_rank"] - group["median_rank"],
                ]
            )
            axes[1, 1].errorbar(
                group["median_rank"],
                group["top_k_frequency"],
                xerr=xerr,
                fmt="o",
                label=(
                    f"{_short_dataset_labels(pd.Series([dataset])).iloc[0]} | "
                    f"{display_methods.get(str(method), method)} | {resource_mode} | "
                    f"{rank_scope} | {level}"
                ),
                alpha=0.75,
                markersize=4,
                capsize=2,
            )
        axes[1, 1].axhline(0.8, color=BLACK, linestyle="--", linewidth=0.8)
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_xlim(left=0.9)
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[1, 1].set_xlabel("Conditional median bootstrap rank (log scale)")
        axes[1, 1].set_ylabel("Tie-inclusive top-k frequency (all replicates)")
        axes[1, 1].legend(fontsize=5)
    axes[1, 1].set_title("Average-rank interval and top-k frequency")
    _panel_label(axes[1, 1], "D")

    figure.suptitle("Real-data ranking stability beyond Spearman and one top-k cutoff")
    figure.subplots_adjust(
        left=0.27, right=0.97, top=0.91, bottom=0.09, hspace=0.35, wspace=0.42
    )
    _save_figure(figure, figures_dir, "figure08_rank_stability")


def _markdown_table(
    table: pd.DataFrame, columns: Sequence[str], limit: int = 30
) -> str:
    available = [column for column in columns if column in table]
    if not available:
        return "_No structured records available._"
    shown = table.loc[:, available].head(limit).copy()
    shown = shown.map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(available) + " |"
    rule = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    suffix = (
        "\n\n_Table truncated in report; complete source data are exported._"
        if len(table) > limit
        else ""
    )
    return "\n".join([header, rule, *rows]) + suffix


def _report_markdown(
    *,
    generated_at: str,
    design: pd.DataFrame,
    coverage: pd.DataFrame,
    loso: pd.DataFrame,
    biology: pd.DataFrame,
    simulation: pd.DataFrame,
    iteration: pd.DataFrame,
    ranking_agreement: pd.DataFrame,
    ranking_tiers: pd.DataFrame,
    adapter_records: pd.DataFrame,
) -> str:
    observed_loso = int(loso["status"].isin(OBSERVED_STATUSES).sum())
    ne_loso = len(loso) - observed_loso
    return f"""# CRYCHIC multi-condition communication benchmark

Generated: `{generated_at}`

Report schema: `{REPORT_SCHEMA_VERSION}`

## Scope and statistical guardrails

This report compares method outputs only within compatible analysis tracks and
resource arms. The primary real-data endpoint is subject-level differential-rank
stability: paired cohorts use leave-one-subject-out reproducibility, while
independent groups use equal-subject repeated split-half stability plus
leave-one-subject influence diagnostics. The MS cross-method endpoint is an
unadjusted CA-vs-Ctrl descriptive comparison; the recorded `~ batch +
lesion_type` formula applies to the CRYCHIC response layer, not to a shared
batch-adjusted endpoint across all methods. Real cohorts do not provide complete edge-level ground
truth, so this report does **not** calculate real-data AUROC/AUPRC and does not
interpret cross-method concordance as accuracy. Missing, failed,
resource-unavailable, and not-estimable states remain explicit; they are never
converted to zero scores.

The NicheNet-labelled Track B output is only a frozen-prior, source-agnostic
ligand-target proxy diagnostic, not native `predict_ligand_activities`. It makes
no sender or ligand-receptor edge claim and is never entered into Track A
concordance. The preregistered Track B primary endpoint, signed target-program
macro-AUPRC from comparable native target predictors, was **not completed** in
this run and remains `NE`; the proxy must not be described as Track B recovery.

Observed primary system rows: **{observed_loso}**; explicit NE rows: **{ne_loso}**.

## Dataset design and estimability

![Figure 1](figures/figure01_design_estimability.png)

**Figure 1.** Dataset scale, biological-unit support, design dimensions, and
estimability status. `NE` entries retain their reason codes.

{_markdown_table(design, ("dataset", "n_cells", "n_samples", "n_subjects", "n_contexts", "design_type", "status", "reason_code"))}

## Resource coverage and run status

![Figure 2](figures/figure02_coverage_status.png)

**Figure 2.** Frozen-resource coverage, comparison-eligible output coverage,
explicit adapter row states, and method/resource completion status. Resource
absence is not scored as a biological negative. Comparison coverage uses each
method/resource arm's own frozen universe and is not database-breadth coverage
across methods. Compact labels use `H` for H-common, `N` for native LR and `P`
for the source-agnostic proxy diagnostic.

{_markdown_table(coverage, ("dataset", "method", "resource_mode", "resource_coverage_fraction", "comparison_coverage_fraction", "status", "reason_code"))}

## Primary real-data endpoint

![Figure 3](figures/figure03_loso_primary.png)

**Figure 3.** Family-macro, subject-level differential-rank stability. Paired
datasets use LOSO with a subject bootstrap interval; independent groups use
equal-subject repeated split-half empirical intervals. The MS comparison is not
cross-method batch-adjusted; its unbalanced batch distribution may contribute
to CA-vs-Ctrl effects. Concordance among methods is not a truth label.

{_markdown_table(loso, ("dataset", "method", "resource_mode", "contrast", "estimate", "ci_lower", "ci_upper", "n_subjects_estimable", "status", "reason_code"))}

## Stability, concordance, and computational cost

![Figure 4](figures/figure04_robustness_performance.png)

**Figure 4.** Real-data, dataset-specific stability and cross-method effect
concordance, followed by descriptive wall time and peak RSS at recorded thread
caps. Native and H-common arms remain labelled; no values are overwritten or
averaged across datasets. Missing-memory states remain `NA`.

## Supportive biology

![Figure 5](figures/figure05_supportive_biology.png)

**Figure 5.** Preregistered literature-supported observations. These are a
supportive silver standard, not comprehensive edge truth and not a basis for
real-data AUROC. Cells with reverse-direction components are annotated with the
number of `opp` components, including mixed partial-support rows.

{_markdown_table(biology, ("dataset", "observation_id", "expected_direction", "method", "resource_mode", "rank_scope", "evidence_rank_scope", "support_status", "observed_direction", "n_components_strong", "n_components_directional", "n_components_opposite", "source_biology_file", "status", "reason_code"))}

## Synthetic and perturbation truth

![Figure 6](figures/figure06_simulation_truth.png)

**Figure 6.** Track A reports active differential LR recovery separately for the
primary oriented-native estimand and the rank-strength sensitivity estimand;
coverage is shown beside AUROC and average precision. Track B is a separate,
source-agnostic frozen-prior receiver-program proxy diagnostic, not native
NicheNet `predict_ligand_activities`, not sender-resolved LR-edge recovery, and
not the preregistered signed target-program macro-AUPRC primary endpoint. That
primary endpoint remains `NE`. All truth-based metrics are confined to explicit
simulation or perturbation scope. The active Track A fixture contains one
positive CXCL10-CXCR3 edge and 44 negatives, so AUROC/AP are a single-positive
recovery diagnostic rather than broad edge-discrimination or generalization
evidence.

{_markdown_table(simulation, ("dataset", "scenario", "method", "resource_mode", "estimand", "record_source", "truth_scope", "metric", "estimate", "status", "reason_code"))}

## Iteration audit

![Figure 7](figures/figure07_iteration_comparison.png)

**Figure 7.** Versioned before/after values for controlled, observed comparisons
and direction-aware relative changes. A negative change is retained as a
regression. Concurrent wall-time and RSS comparisons are not scored and appear
as `NE` with their reason codes.

{_markdown_table(iteration, ("dataset", "method", "metric", "iteration_from", "iteration_to", "before", "after", "metric_direction", "signed_improvement", "change_class", "status", "reason_code"))}

## Top-sensitive ranking stability

![Figure 8](figures/figure08_rank_stability.png)

**Figure 8.** Subject split-half rank-biased overlap (`p=0.9`), top-weighted
Kendall tau (`power=1`), preregistered LR-family top-k curves (`k=1..25`) and LR curves
(`k=1..100`), plus 2,000-subject-bootstrap average-rank intervals, rank
availability, tie-inclusive top-k frequency, and stable tiers (`95%` quantile
intervals; minimum top-k frequency `0.80`; seed `20260712`). Rank availability
is a missingness diagnostic, not model/family selection frequency; top-k
frequency uses all requested replicates, including non-estimable replicates, in
its denominator. NicheNet Track B does not support LR or sender
rank claims, and unsupported Kuppe/PancVAX designs remain explicit `NE`. The
current score contract has no frozen LR equivalence/driver-family identifier,
so `lr_family` remains `NE`; observed `sender_receiver_pair` rows are labelled
separately and are not represented as mechanistic LR families.

{_markdown_table(ranking_agreement, ("dataset", "method", "resource_mode", "rank_scope", "ranking_level", "metric", "estimate", "envelope_lower", "envelope_upper", "n_repeats_estimable", "status", "reason_code"))}

{_markdown_table(ranking_tiers, ("dataset", "method", "resource_mode", "rank_scope", "ranking_level", "item_label", "tier", "rank_availability_frequency", "top_k_frequency", "lower_rank", "upper_rank", "status", "reason_code"))}

## Adapter inventory

{_markdown_table(adapter_records, ("dataset", "method", "method_version", "resource", "status", "reason_code", "path"))}

## Reproducibility artifacts

- `input_manifest.tsv`: every consumed input with byte size and SHA256.
- `artifact_manifest.tsv`: every generated report artifact with byte size and SHA256.
- `report_manifest.json`: report contract, guardrails, environment, and artifact inventory.
- `source_data/figure*.csv`: the exact table passed to each figure.
- `figures/*.png`: 300 dpi bitmap output.
- `figures/*.svg` and `figures/*.pdf`: vector exports from the same figure object.
- `REPORT.html`: self-contained HTML with embedded figure images and CSS.
- `REPORT.pdf`: paginated report rendered from the self-contained HTML.
"""


def _html_table(table: pd.DataFrame, columns: Sequence[str], limit: int = 30) -> str:
    available = [column for column in columns if column in table]
    if not available:
        return "<p><em>No structured records available.</em></p>"
    shown = table.loc[:, available].head(limit)
    head = "".join(f"<th>{html.escape(column)}</th>" for column in available)
    body_rows: list[str] = []
    for row in shown.itertuples(index=False, name=None):
        cells = "".join(
            f"<td>{html.escape('' if pd.isna(value) else str(value))}</td>"
            for value in row
        )
        body_rows.append(f"<tr>{cells}</tr>")
    suffix = (
        "<p><em>Table truncated in report; complete source data are exported.</em></p>"
        if len(table) > limit
        else ""
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>{suffix}"


def _image_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _report_html(
    *,
    generated_at: str,
    css: str,
    figures_dir: Path,
    design: pd.DataFrame,
    coverage: pd.DataFrame,
    loso: pd.DataFrame,
    biology: pd.DataFrame,
    simulation: pd.DataFrame,
    iteration: pd.DataFrame,
    ranking_agreement: pd.DataFrame,
    ranking_tiers: pd.DataFrame,
    adapter_records: pd.DataFrame,
) -> str:
    figures = [
        (
            "Dataset design and estimability",
            "figure01_design_estimability.png",
            "Dataset scale, biological replication, and explicit estimability states.",
        ),
        (
            "Resource coverage and status",
            "figure02_coverage_status.png",
            "Arm-specific comparison eligibility and adapter states; not a "
            "cross-method database-breadth comparison.",
        ),
        (
            "Primary real-data endpoint",
            "figure03_loso_primary.png",
            "Design-appropriate subject-level differential-rank stability with intervals.",
        ),
        (
            "Robustness and performance",
            "figure04_robustness_performance.png",
            "Dataset-specific real-data stability, descriptive runtime, and memory.",
        ),
        (
            "Supportive biology",
            "figure05_supportive_biology.png",
            "Preregistered supportive observations; not complete edge truth.",
        ),
        (
            "Synthetic truth",
            "figure06_simulation_truth.png",
            "Track A is a one-positive/44-negative LR diagnostic; the Track B "
            "proxy is separate and its preregistered primary endpoint remains NE.",
        ),
        (
            "Iteration audit",
            "figure07_iteration_comparison.png",
            "Only observed controlled comparisons are scored; uncontrolled timing is NE.",
        ),
        (
            "Top-sensitive ranking stability",
            "figure08_rank_stability.png",
            "RBO, weighted Kendall, complete top-k curves, bootstrap rank "
            "intervals, selection frequency, and stable tiers; unsupported "
            "rank claims remain NE.",
        ),
    ]
    figure_html = "".join(
        f"<h2>{html.escape(title)}</h2><figure>"
        f'<img src="{_image_data_uri(figures_dir / file_name)}" '
        f'alt="{html.escape(title)}"><figcaption>{html.escape(caption)}</figcaption>'
        "</figure>"
        for title, file_name, caption in figures
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CRYCHIC multi-condition benchmark</title><style>{css}</style></head>
<body><h1>CRYCHIC multi-condition communication benchmark</h1>
<p>Generated: <code>{html.escape(generated_at)}</code></p>
<h2>Scope and guardrails</h2>
<p>The primary real-data endpoint is subject-level differential-rank stability:
paired cohorts use leave-one-subject-out reproducibility, while independent
groups use stratified split-half stability and leave-one-subject influence.
The MS cross-method endpoint is an equal-subject, unadjusted descriptive
CA-vs-Ctrl comparison; the batch formula is not shared by all methods.
Real-data AUROC/AUPRC are not calculated. Cross-method concordance is not
accuracy. Missing, failed, unavailable, and not-estimable states remain explicit
and are never converted to zero.</p>
<p>The NicheNet-labelled Track B output is a frozen-prior, source-agnostic
ligand-target proxy diagnostic rather than native
<code>predict_ligand_activities</code>. It makes no sender or ligand-receptor
edge claim and is not mixed into Track A concordance. The preregistered signed
target-program macro-AUPRC primary endpoint was not completed and remains
<code>NE</code>.</p>
{figure_html}
<h2>Dataset design table</h2>
{_html_table(design, ("dataset", "n_cells", "n_samples", "n_subjects", "n_contexts", "design_type", "status", "reason_code"))}
<h2>Coverage table</h2>
{_html_table(coverage, ("dataset", "method", "resource_mode", "resource_coverage_fraction", "comparison_coverage_fraction", "status", "reason_code"))}
<h2>Primary subject-level endpoint</h2>
{_html_table(loso, ("dataset", "method", "resource_mode", "contrast", "estimate", "ci_lower", "ci_upper", "n_subjects_estimable", "status", "reason_code"))}
<h2>Supportive biology table</h2>
{_html_table(biology, ("dataset", "observation_id", "expected_direction", "method", "resource_mode", "rank_scope", "evidence_rank_scope", "support_status", "n_components_strong", "n_components_directional", "n_components_opposite", "source_biology_file", "status", "reason_code"))}
<h2>Simulation truth table</h2>
{_html_table(simulation, ("dataset", "scenario", "method", "resource_mode", "estimand", "record_source", "truth_scope", "metric", "estimate", "status", "reason_code"))}
<h2>Iteration comparison</h2>
{_html_table(iteration, ("dataset", "method", "metric", "before", "after", "metric_direction", "signed_improvement", "change_class", "status", "reason_code"))}
<h2>Top-sensitive ranking agreement</h2>
{_html_table(ranking_agreement, ("dataset", "method", "resource_mode", "rank_scope", "ranking_level", "metric", "estimate", "envelope_lower", "envelope_upper", "n_repeats_estimable", "status", "reason_code"))}
<h2>Bootstrap-supported stable tiers</h2>
{_html_table(ranking_tiers, ("dataset", "method", "resource_mode", "rank_scope", "ranking_level", "item_label", "tier", "rank_availability_frequency", "top_k_frequency", "lower_rank", "upper_rank", "status", "reason_code"))}
<h2>Adapter inventory</h2>
{_html_table(adapter_records, ("dataset", "method", "method_version", "resource", "status", "reason_code", "path"))}
</body></html>"""


def _render_pdf(html_path: Path, pdf_path: Path) -> None:
    executable = shutil.which("weasyprint")
    if executable is None:
        raise RuntimeError("REPORT.pdf rendering requires the weasyprint executable")
    result = subprocess.run(
        [executable, str(html_path), str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"weasyprint failed: {detail}")


def _input_manifest(
    inputs: ReportInputs,
    workspace_root: Path,
    extra_paths: Sequence[tuple[Path, str]] = (),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path, role in [*inputs.paths_with_roles(), *extra_paths]:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            display = resolved.relative_to(workspace_root.resolve()).as_posix()
        except ValueError:
            display = str(resolved)
        rows.append(
            {
                "path": display,
                "role": role,
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
                "status": "verified_present",
            }
        )
    return pd.DataFrame(rows).sort_values("path", kind="stable", ignore_index=True)


def _artifact_records(
    output: Path, paths_with_roles: Sequence[tuple[Path, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, role in paths_with_roles:
        resolved = path.resolve()
        rows.append(
            {
                "path": resolved.relative_to(output.resolve()).as_posix(),
                "role": role,
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
                "status": "verified_present",
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def generate(
    final_dir: Path,
    output_dir: Path,
    generated_at: str,
    *,
    input_spec: Path | None = None,
    render_pdf: bool = True,
    report_id: str | None = None,
) -> None:
    """Generate all report artifacts from a frozen multi-condition result root."""

    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    default_truth = (
        repo_root / "benchmarks/truth/multicondition_supportive_biology.yaml"
    )
    inputs = resolve_inputs(
        final_dir,
        input_spec=input_spec,
        default_truth=default_truth,
    )
    output = (
        output_dir.resolve() if output_dir.is_absolute() else repo_root / output_dir
    )
    resolved_report_id = (report_id or output.name).strip()
    if not resolved_report_id:
        raise ValueError("report_id must not be empty")
    figures_dir = output / "figures"
    source_dir = output / "source_data"
    output.mkdir(parents=True, exist_ok=True)

    style_path = repo_root / "benchmarks/report/publication.mplstyle"
    plt.style.use(style_path)
    tables = {key: _read_table(path) for key, path in inputs.metric_tables.items()}
    design = _design_source(tables["dataset_design"], inputs.dataset_manifests)
    coverage = _coverage_source(tables["coverage"], inputs.score_tables)
    coverage = coverage.loc[
        coverage["truth_scope"].eq("real_data") | coverage["method"].eq("NE")
    ].reset_index(drop=True)
    if coverage.empty:
        coverage = _ne_source(reason_code="real_data_coverage_missing")
    loso = _loso_source(tables["loso_primary"])
    stability = _stability_source(tables["stability"])
    concordance = _concordance_source(tables["concordance"])
    performance = _performance_source(tables["performance"])
    biology = _biology_source(
        tables["biology_support"],
        inputs.truth_yaml,
        inputs.supportive_biology_dataset_aliases,
    )
    simulation = _simulation_source(tables["simulation_truth"])
    iteration = _iteration_source(tables["iteration_comparison"])
    ranking_agreement = _ranking_agreement_source(tables["ranking_agreement"])
    ranking_curve = _ranking_curve_source(tables["ranking_top_k_curve"])
    ranking_intervals = _ranking_intervals_source(tables["ranking_intervals"])
    ranking_tiers = _ranking_tiers_source(tables["ranking_tiers"])

    real_stability = stability.loc[
        stability["truth_scope"].eq("real_data") | stability["method"].eq("NE")
    ].copy()
    real_concordance = concordance.loc[
        concordance["truth_scope"].eq("real_data")
        | concordance["method_left"].eq("NE")
    ].copy()
    real_performance = performance.loc[
        performance["truth_scope"].eq("real_data") | performance["method"].eq("NE")
    ].copy()
    robustness_source = pd.concat(
        [
            real_stability.assign(panel="A_stability"),
            real_concordance.assign(panel="B_concordance"),
            real_performance.assign(panel="C_D_performance"),
        ],
        ignore_index=True,
        sort=False,
    )
    ranking_source = pd.concat(
        [
            ranking_agreement.assign(panel="A_agreement"),
            ranking_curve.assign(panel="B_top_k_curve"),
            ranking_tiers.assign(panel="C_stable_tiers"),
            ranking_intervals.assign(panel="D_rank_intervals"),
        ],
        ignore_index=True,
        sort=False,
    )
    figure_sources = {
        "figure01_design_estimability": design,
        "figure02_coverage_status": coverage,
        "figure03_loso_primary": loso,
        "figure04_robustness_performance": robustness_source,
        "figure05_supportive_biology": biology,
        "figure06_simulation_truth": simulation,
        "figure07_iteration_comparison": iteration,
        "figure08_rank_stability": ranking_source,
    }
    for stem, source in figure_sources.items():
        _save_source(source, source_dir, stem)

    _plot_design(design, figures_dir)
    _plot_coverage(coverage, figures_dir)
    _plot_loso(loso, figures_dir)
    _plot_robustness(
        real_stability, real_concordance, real_performance, figures_dir
    )
    _plot_biology(biology, figures_dir)
    _plot_simulation(simulation, figures_dir)
    _plot_iteration(iteration, figures_dir)
    _plot_rank_stability(
        ranking_agreement,
        ranking_curve,
        ranking_intervals,
        ranking_tiers,
        figures_dir,
    )

    adapter_records = pd.DataFrame(_load_manifests(inputs.adapter_manifests))
    if adapter_records.empty:
        adapter_records = _ne_source(reason_code="adapter_manifests_missing")
        adapter_records["path"] = ""
    provenance_candidates = (
        (inputs.final_dir / "finalization_manifest.json", "finalization manifest"),
        (inputs.final_dir / "SHA256SUMS.tsv", "finalizer checksum manifest"),
        (
            inputs.final_dir / "provenance/run_specification.json",
            "frozen finalizer run specification",
        ),
        (
            repo_root / "benchmarks/metrics/finalize_multicondition.py",
            "finalizer source",
        ),
        (
            repo_root / "benchmarks/metrics/multicondition.py",
            "metric implementation source",
        ),
        (
            repo_root / "benchmarks/metrics/multicondition_rank_stability.py",
            "top-sensitive ranking metric integration source",
        ),
        (repo_root / "pyproject.toml", "Python project specification"),
        (repo_root / "uv.lock", "Python dependency lock"),
        (repo_root / "DEVELOPMENT_PLAN.md", "development gate specification"),
        (workspace_root / "benchmark.md", "benchmark request specification"),
    )
    input_manifest = _input_manifest(
        inputs,
        workspace_root,
        extra_paths=(
            (Path(__file__).resolve(), "report generator source"),
            (style_path, "publication figure style"),
            (repo_root / "benchmarks/report/report.css", "report HTML/PDF style"),
            (
                repo_root
                / "benchmarks/literature/MULTICONDITION_BENCHMARK_PROTOCOL.md",
                "preregistered benchmark protocol",
            ),
            *(item for item in provenance_candidates if item[0].is_file()),
        ),
    )
    input_manifest.to_csv(output / "input_manifest.tsv", sep="\t", index=False)

    markdown = _report_markdown(
        generated_at=generated_at,
        design=design,
        coverage=coverage,
        loso=loso,
        biology=biology,
        simulation=simulation,
        iteration=iteration,
        ranking_agreement=ranking_agreement,
        ranking_tiers=ranking_tiers,
        adapter_records=adapter_records,
    )
    (output / "REPORT.md").write_text(markdown, encoding="utf-8")
    css_path = repo_root / "benchmarks/report/report.css"
    css = css_path.read_text(encoding="utf-8").replace(
        "CRYCHIC canonical_v01", "CRYCHIC multi-condition benchmark"
    )
    report_html = _report_html(
        generated_at=generated_at,
        css=css,
        figures_dir=figures_dir,
        design=design,
        coverage=coverage,
        loso=loso,
        biology=biology,
        simulation=simulation,
        iteration=iteration,
        ranking_agreement=ranking_agreement,
        ranking_tiers=ranking_tiers,
        adapter_records=adapter_records,
    )
    html_path = output / "REPORT.html"
    html_path.write_text(report_html, encoding="utf-8")
    if render_pdf:
        _render_pdf(html_path, output / "REPORT.pdf")

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "git_revision": _git_revision(repo_root),
        "git_state": _git_state(repo_root),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "matplotlib": _package_version("matplotlib"),
        "pyarrow": _package_version("pyarrow"),
        "weasyprint": _executable_version("weasyprint"),
        "report_randomness": "none",
    }
    main_figures = (
        "figure01_design_estimability",
        "figure02_coverage_status",
        "figure03_loso_primary",
        "figure04_robustness_performance",
        "figure05_supportive_biology",
        "figure06_simulation_truth",
        "figure07_iteration_comparison",
        "figure08_rank_stability",
    )
    artifact_paths: list[tuple[Path, str]] = [
        (output / "input_manifest.tsv", "report input checksum manifest"),
        (output / "REPORT.md", "Markdown report"),
        (output / "REPORT.html", "self-contained HTML report"),
    ]
    if render_pdf:
        artifact_paths.append((output / "REPORT.pdf", "paginated PDF report"))
    for stem in main_figures:
        artifact_paths.append(
            (source_dir / f"{stem}.csv", f"figure source data: {stem}")
        )
        for suffix in ("png", "svg", "pdf"):
            artifact_paths.append(
                (figures_dir / f"{stem}.{suffix}", f"figure {suffix}: {stem}")
            )
    artifacts = _artifact_records(output, artifact_paths)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": resolved_report_id,
        "generated_at": generated_at,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "environment": environment,
        "guardrails": {
            "real_data_edge_auroc_reported": False,
            "real_data_edge_auprc_reported": False,
            "cross_method_concordance_treated_as_accuracy": False,
            "missing_states_imputed_as_zero": False,
            "ms_cross_method_endpoint_claimed_batch_adjusted": False,
            "single_class_observed_auc_allowed": False,
            "calibration_metrics_without_1000_nulls_allowed": False,
            "real_data_rows_require_explicit_truth_scope": True,
            "preregistered_track_b_primary_reported": False,
            "track_b_proxy_mislabelled_as_native_nichenet": False,
            "nichenet_lr_or_sender_ranking_reported": False,
            "synthetic_truth_scopes": sorted(ALLOWED_SYNTHETIC_SCOPES),
        },
        "ranking_parameters": {
            "rbo_persistence": 0.9,
            "weighted_kendall_power": 1.0,
            "family_top_k_curve": [1, 25],
            "lr_top_k_curve": [1, 100],
            "sender_receiver_pair_top_k_curve": [1, 25],
            "lr_family_mapping_status": "not_available_in_score_contract",
            "n_bootstrap": 2000,
            "n_split_repeats": 200,
            "rank_interval_quantile_level": 0.95,
            "minimum_top_k_frequency": 0.8,
            "minimum_estimable_replicate_fraction": 0.8,
            "minimum_subjects": 3,
            "minimum_observed_ranks": 2,
            "random_seed": 20260712,
            "resampling_unit": "subject",
            "rank_interval_conditioning": "conditional_on_rank_availability",
            "rank_availability_frequency_denominator": "all_requested_replicates",
            "top_k_frequency_denominator": "all_requested_replicates",
            "tie_policy": "average_rank_and_tie_inclusive_top_k",
        },
        "input_manifest": "input_manifest.tsv",
        "artifact_manifest": "artifact_manifest.tsv",
        "artifacts": artifacts,
        "input_count": len(input_manifest),
        "adapter_manifest_count": len(inputs.adapter_manifests),
        "score_table_count": len(inputs.score_tables),
        "dataset_manifest_count": len(inputs.dataset_manifests),
        "metric_tables": {
            key: str(path) if path is not None else None
            for key, path in inputs.metric_tables.items()
        },
        "figures": [
            {
                "id": stem,
                "source_data": f"source_data/{stem}.csv",
                "formats": [
                    f"figures/{stem}.png",
                    f"figures/{stem}.svg",
                    f"figures/{stem}.pdf",
                ],
                "png_dpi": 300,
            }
            for stem in main_figures
        ],
        "report_artifacts": [
            "REPORT.md",
            "REPORT.html",
            *(["REPORT.pdf"] if render_pdf else []),
        ],
        "endpoint_status_counts": {
            "loso_observed": int(loso["status"].isin(OBSERVED_STATUSES).sum()),
            "loso_not_estimable": int((~loso["status"].isin(OBSERVED_STATUSES)).sum()),
            "biology_supported": int(
                biology.get("support_status", pd.Series(dtype=str))
                .eq("supported")
                .sum()
            ),
            "iteration_regressions": int(
                iteration.get("change_class", pd.Series(dtype=str))
                .eq("regressed")
                .sum()
            ),
            "ranking_agreement_observed": int(
                ranking_agreement["status"].isin(OBSERVED_STATUSES).sum()
            ),
            "ranking_agreement_not_estimable": int(
                (~ranking_agreement["status"].isin(OBSERVED_STATUSES)).sum()
            ),
            "stable_top_k_items": int(
                ranking_tiers.get("tier", pd.Series(dtype=str))
                .eq("stable_top_k")
                .sum()
            ),
        },
    }
    report_manifest_path = output / "report_manifest.json"
    report_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_manifest_rows = [
        *artifacts,
        *_artifact_records(
            output, ((report_manifest_path, "report contract manifest"),)
        ),
    ]
    pd.DataFrame(artifact_manifest_rows).to_csv(
        output / "artifact_manifest.tsv", sep="\t", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-dir",
        required=True,
        type=Path,
        help="Frozen multi-condition benchmark result directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/multicondition_v01"),
        help="Absolute path or path relative to the CRYCHIC repository",
    )
    parser.add_argument(
        "--input-spec",
        type=Path,
        default=None,
        help="Optional report_inputs.json; defaults to FINAL_DIR/report_inputs.json",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="ISO-8601 report timestamp; defaults to current local time",
    )
    parser.add_argument(
        "--report-id",
        default=None,
        help="Stable report identifier; defaults to the output directory name",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Development-only: do not render REPORT.pdf",
    )
    args = parser.parse_args()
    generated_at = args.generated_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    generate(
        args.final_dir,
        args.output_dir,
        generated_at,
        input_spec=args.input_spec,
        render_pdf=not args.skip_pdf,
        report_id=args.report_id,
    )


if __name__ == "__main__":
    main()
