"""Generate the canonical-data v0.1 benchmark report and source data."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy import sparse
from scipy.stats import spearmanr

from crychic.resources import load_cellchat_resource

REPORT_ID = "canonical_v01"
SCHEMA_VERSION = "canonical_metrics_summary.v1"

# Okabe-Ito colors, supplemented with neutral grays.
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
PALETTE = [BLUE, VERMILLION, GREEN, ORANGE, PINK, SKY, BLACK]

ISG_GENES = [
    "IFI6",
    "IFI44",
    "IFI44L",
    "IFIT1",
    "IFIT2",
    "IFIT3",
    "ISG15",
    "MX1",
    "OAS1",
    "OAS2",
    "STAT1",
    "STAT2",
]
ANTIGEN_GENES = ["B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2"]
TF_GENES = ["ASCL2", "GATA3", "STAT1", "STAT2", "HIF1A"]


def _native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MetricBook:
    """Collect a stable, flat metric table for TSV and JSON export."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        *,
        section: str,
        dataset: str,
        metric: str,
        value: float | int | None = None,
        value_text: str | None = None,
        unit: str = "",
        status: str,
        reason_code: str = "",
        method: str = "CRYCHIC preliminary",
        resource: str = "",
        interpretation: str = "",
        source_figure: str = "",
    ) -> None:
        self.rows.append(
            {
                "section": section,
                "dataset": dataset,
                "method": method,
                "resource": resource,
                "metric": metric,
                "value_numeric": _native(value),
                "value_text": value_text or "",
                "unit": unit,
                "status": status,
                "reason_code": reason_code,
                "interpretation": interpretation,
                "source_figure": source_figure,
            }
        )

    def frame(self) -> pd.DataFrame:
        columns = [
            "section",
            "dataset",
            "method",
            "resource",
            "metric",
            "value_numeric",
            "value_text",
            "unit",
            "status",
            "reason_code",
            "interpretation",
            "source_figure",
        ]
        return pd.DataFrame(self.rows, columns=columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _relative(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _input_record(path: Path, workspace_root: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path, workspace_root),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "status": "verified_present",
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _tool_version(executable: str, *arguments: str) -> str:
    path = shutil.which(executable)
    if path is None:
        return "not_installed"
    result = subprocess.run(
        [path, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if result.returncode == 0 and output else "unavailable"


def _render_report(output_dir: Path, css_path: Path) -> None:
    pandoc = shutil.which("pandoc")
    weasyprint = shutil.which("weasyprint")
    missing = [
        name
        for name, path in (("pandoc", pandoc), ("weasyprint", weasyprint))
        if path is None
    ]
    if missing:
        raise RuntimeError(
            "report rendering requires installed tools: " + ", ".join(missing)
        )
    commands = [
        [
            str(pandoc),
            "REPORT.md",
            "--from=gfm",
            "--to=html5",
            "--standalone",
            "--embed-resources",
            "--resource-path=.",
            f"--css={css_path}",
            "--metadata=lang:zh-CN",
            "--metadata=title:CRYCHIC canonical_v01 benchmark report",
            "--toc",
            "--toc-depth=2",
            "--output=REPORT.html",
        ],
        [str(weasyprint), "REPORT.html", "REPORT.pdf"],
    ]
    for command in commands:
        try:
            subprocess.run(
                command,
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown renderer error").strip()
            raise RuntimeError(
                f"report renderer {Path(command[0]).name} failed: {detail}"
            ) from exc


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


def _save_figure(figure: Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        figure.savefig(figures_dir / f"{stem}.{suffix}", facecolor="white")
    figure.savefig(figures_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)


def _save_source(data: pd.DataFrame, source_dir: Path, stem: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(source_dir / f"{stem}.csv", index=False, na_rep="")


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _overview_data(
    benchmark_root: Path,
    metrics: dict[str, dict[str, Any]],
    embryo_composition: pd.DataFrame,
    book: MetricBook,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    final_inputs: dict[str, dict[str, Any]] = {}
    for key, final_name in (
        ("kang", "kang2018"),
        ("ad", "ad_skin"),
        ("trophoblast", "trophoblast"),
    ):
        path = benchmark_root / "canonical_v01_results" / final_name / "metrics.json"
        final_inputs[key] = _read_json(path).get("input", {}) if path.is_file() else {}
    dataset_specs = [
        {
            "dataset": "Kang IFN-beta",
            "key": "kang",
            "cells": int(metrics["kang"]["input_shape"][0]),
            "analysis_units": int(metrics["kang"]["n_samples"]),
            "subjects": int(metrics["kang"]["n_subjects"]),
            "contexts": 2,
            "cell_types": 8,
            "final_analysis_cells": (
                final_inputs["kang"].get("analysis_shape") or [None]
            )[0],
            "final_analysis_units": final_inputs["kang"].get("n_samples"),
            "final_subjects": final_inputs["kang"].get("n_subjects"),
            "final_cell_types": final_inputs["kang"].get("n_cell_types"),
            "input_mode": "raw counts",
            "design_status": "paired_subject_design",
        },
        {
            "dataset": "AD skin",
            "key": "ad",
            "cells": int(metrics["ad"]["input_shape"][0]),
            "analysis_units": int(metrics["ad"]["n_samples"]),
            "subjects": int(metrics["ad"]["n_subjects"]),
            "contexts": 2,
            "cell_types": 12,
            "final_analysis_cells": (
                final_inputs["ad"].get("analysis_shape") or [None]
            )[0],
            "final_analysis_units": final_inputs["ad"].get("n_samples"),
            "final_subjects": final_inputs["ad"].get("n_subjects"),
            "final_cell_types": final_inputs["ad"].get("n_cell_types"),
            "input_mode": "normalized only",
            "design_status": "unpaired_exploratory",
        },
        {
            "dataset": "Trophoblast",
            "key": "trophoblast",
            "cells": int(metrics["trophoblast"]["input_shape"][0]),
            "analysis_units": int(metrics["trophoblast"]["n_samples"]),
            "subjects": int(metrics["trophoblast"]["n_subjects"]),
            "contexts": 1,
            "cell_types": 42,
            "final_analysis_cells": (
                final_inputs["trophoblast"].get("analysis_shape") or [None]
            )[0],
            "final_analysis_units": final_inputs["trophoblast"].get("n_samples"),
            "final_subjects": final_inputs["trophoblast"].get("n_subjects"),
            "final_cell_types": final_inputs["trophoblast"].get("n_cell_types"),
            "input_mode": "raw counts",
            "design_status": "single_context_exploratory",
        },
        {
            "dataset": "Embryonic skin",
            "key": "embryo",
            "cells": int(embryo_composition["n_cells"].sum()),
            "analysis_units": 2,
            "subjects": None,
            "contexts": 2,
            "cell_types": int(embryo_composition["cell_type"].nunique()),
            "final_analysis_cells": None,
            "final_analysis_units": None,
            "final_subjects": None,
            "final_cell_types": None,
            "input_mode": "normalized reference",
            "design_status": "merged_stage_objects_no_replicates",
        },
    ]
    source_rows: list[dict[str, Any]] = []
    for spec in dataset_specs:
        dataset = str(spec["dataset"])
        for metric_name, unit in (
            ("cells", "cells"),
            ("analysis_units", "samples_or_objects"),
            ("subjects", "biological_subjects"),
            ("contexts", "contexts"),
            ("cell_types", "cell_types"),
            ("final_analysis_cells", "cells"),
            ("final_analysis_units", "samples"),
            ("final_subjects", "biological_subjects"),
            ("final_cell_types", "cell_types"),
        ):
            value = spec[metric_name]
            source_rows.append(
                {
                    "panel": (
                        "A" if metric_name in {"cells", "final_analysis_cells"} else "B"
                    ),
                    "dataset": dataset,
                    "metric": metric_name,
                    "value": value,
                    "unit": unit,
                    "status": (
                        "not_available"
                        if value is None
                        else (
                            "final_analyzed_subset"
                            if metric_name.startswith("final_")
                            else str(spec["design_status"])
                        )
                    ),
                    "resource": "",
                }
            )
            book.add(
                section="dataset_overview",
                dataset=dataset,
                metric=metric_name,
                value=value,
                unit=unit,
                status=("not_available" if value is None else "descriptive"),
                reason_code=("no_subject_level_replicates" if value is None else ""),
                interpretation=str(spec["design_status"]),
                source_figure="figure01_overview",
            )

    for final_name, dataset, resource in (
        ("kang2018", "Kang IFN-beta", "CellChatDB human v2"),
        ("ad_skin", "AD skin", "CellChatDB human v2"),
        ("trophoblast", "Trophoblast", "CellPhoneDB human v5.0.0"),
    ):
        final_root = benchmark_root / "canonical_v01_results" / final_name
        final_metrics = _read_json(final_root / "metrics.json")
        if final_metrics.get("status") != "complete":
            raise ValueError(f"Final resource audit is incomplete for {final_name}")
        dry_run_resources = final_metrics.get("dry_run", {}).get("resources", [])
        lr_resource = next(
            (
                item
                for item in dry_run_resources
                if item.get("resource_kind") == "lr_bundle"
            ),
            None,
        )
        if lr_resource is None:
            raise ValueError(
                f"Final dry-run LR resource audit is missing for {final_name}"
            )
        source = int(lr_resource["source_entities"])
        mapped = int(lr_resource["mapped_entities"])
        final_interactions = pd.read_parquet(
            final_root / "result/interactions.parquet", columns=["interaction_id"]
        )
        selected = int(final_interactions["interaction_id"].nunique())
        selection = final_metrics.get("interaction_selection") or {}
        truncated = bool(selection.get("resource_truncated", False))
        for metric_name, value in (
            ("gene_mapped_fraction", mapped / source),
            ("final_selected_fraction", selected / source),
        ):
            if metric_name == "gene_mapped_fraction":
                status = "final_dry_run_ready"
            else:
                status = (
                    "resource_truncated_v0_1_smoke"
                    if truncated
                    else "final_selected_full_resource"
                )
            source_rows.append(
                {
                    "panel": "C",
                    "dataset": dataset,
                    "metric": metric_name,
                    "value": value,
                    "unit": "fraction_of_source_interactions",
                    "status": status,
                    "resource": resource,
                    "source_interactions": source,
                    "gene_mapped_interactions": mapped,
                    "final_selected_interactions": selected,
                    "selection_rule": selection.get("selection_rule", "not_available"),
                }
            )
            book.add(
                section="resource_mapping",
                dataset=dataset,
                metric=metric_name,
                value=value,
                unit="fraction",
                status=status,
                method="CRYCHIC canonical v0.1 runner",
                resource=resource,
                interpretation=(
                    f"source={source}; mapped={mapped}; final_selected={selected}; "
                    f"selection_rule={selection.get('selection_rule', 'not_available')}"
                ),
                source_figure="figure01_overview",
            )

    phases = ["validate", "pseudobulk", "resource_load", "response", "availability"]
    for key, dataset in (
        ("kang", "Kang IFN-beta"),
        ("ad", "AD skin"),
        ("trophoblast", "Trophoblast"),
    ):
        timing = metrics[key]["timing_seconds"]
        for phase in phases:
            value = float(timing[phase])
            source_rows.append(
                {
                    "panel": "D",
                    "dataset": dataset,
                    "metric": phase,
                    "value": value,
                    "unit": "seconds",
                    "status": "observed_runtime",
                    "resource": "",
                }
            )
        preliminary_total = float(timing["total_compute"])
        source_rows.append(
            {
                "panel": "D",
                "dataset": dataset,
                "metric": "preliminary_total_compute",
                "value": preliminary_total,
                "unit": "seconds",
                "status": "observed_preliminary_runtime",
                "resource": "",
                "peak_rss_mb": None,
                "peak_rss_scope": "not_recorded",
            }
        )
        final_name = {
            "kang": "kang2018",
            "ad": "ad_skin",
            "trophoblast": "trophoblast",
        }[key]
        final_metrics_path = (
            benchmark_root / "canonical_v01_results" / final_name / "metrics.json"
        )
        final_wall: float | None = None
        final_peak: float | None = None
        final_status = "pending_final_run"
        peak_scope = "not_available"
        if final_metrics_path.is_file():
            final_metrics = _read_json(final_metrics_path)
            final_status = str(final_metrics.get("status", "unknown"))
            if final_status == "complete":
                final_wall = float(final_metrics["timing_seconds"]["wall"])
                final_peak = float(final_metrics["peak_rss_mb"])
                memory = final_metrics.get("memory_measurement") or {}
                isolated = final_metrics.get(
                    "isolated_process", memory.get("isolated_process")
                )
                peak_scope = str(
                    memory.get("peak_rss_scope")
                    or final_metrics.get("peak_rss_scope")
                    or (
                        "standalone_isolated_process"
                        if isolated is True
                        else "sequential_process_upper_bound"
                    )
                )
        source_rows.append(
            {
                "panel": "D",
                "dataset": dataset,
                "metric": "canonical_final_wall",
                "value": final_wall,
                "unit": "seconds",
                "status": final_status,
                "resource": "dataset_specific_final_resource",
                "peak_rss_mb": final_peak,
                "peak_rss_scope": peak_scope,
            }
        )
        book.add(
            section="runtime",
            dataset=dataset,
            metric="total_compute_seconds",
            value=preliminary_total,
            unit="seconds",
            status="observed_runtime",
            reason_code="runtime_hardware_not_recorded",
            interpretation="Preliminary wall-clock timing; hardware was not persisted.",
            source_figure="figure01_overview",
        )
    return pd.DataFrame(source_rows), dataset_specs


def _plot_overview(source: pd.DataFrame, figures_dir: Path) -> None:
    order = ["Kang IFN-beta", "AD skin", "Trophoblast", "Embryonic skin"]
    colors = dict(zip(order, [BLUE, VERMILLION, GREEN, ORANGE], strict=True))
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))

    axis = axes[0, 0]
    data = source[(source["panel"] == "A") & (source["metric"] == "cells")]
    values = [
        float(data.loc[data["dataset"] == item, "value"].iloc[0]) for item in order
    ]
    bars = axis.barh(
        order[::-1], values[::-1], color=[colors[item] for item in order[::-1]]
    )
    final_data = source[
        (source["panel"] == "A") & (source["metric"] == "final_analysis_cells")
    ]
    final_values = []
    for item in order:
        value = final_data.loc[final_data["dataset"] == item, "value"].iloc[0]
        final_values.append(float(value) if pd.notna(value) else np.nan)
    finite = np.isfinite(final_values)
    axis.scatter(
        np.asarray(final_values)[finite][::-1],
        np.arange(len(order))[finite[::-1]],
        color=BLACK,
        marker="D",
        s=28,
        zorder=4,
    )
    axis.set_xscale("log")
    axis.set_xlabel("Cells (log scale)")
    axis.set_title("Source/preliminary scale and final analyzed subset")
    for bar, value in zip(bars, values[::-1], strict=True):
        axis.text(
            value * 1.06,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            fontsize=7,
        )
    final_types = source[
        (source["panel"] == "B") & (source["metric"] == "final_cell_types")
    ].set_index("dataset")["value"]
    for index, (dataset, value) in enumerate(
        zip(order[::-1], final_values[::-1], strict=True)
    ):
        if np.isfinite(value) and value != values[::-1][index]:
            types = final_types.get(dataset)
            axis.text(
                value * 1.10,
                index,
                f"{value:,.0f} / {int(types)} types",
                ha="left",
                va="center",
                fontsize=6,
                color=BLACK,
            )
    _panel_label(axis, "A")

    axis = axes[0, 1]
    data = source[
        (source["panel"] == "B") & source["metric"].isin(["analysis_units", "subjects"])
    ]
    x = np.arange(len(order))
    units = [
        float(
            data[(data["dataset"] == item) & (data["metric"] == "analysis_units")][
                "value"
            ].iloc[0]
        )
        for item in order
    ]
    subjects = []
    for item in order:
        value = data[(data["dataset"] == item) & (data["metric"] == "subjects")][
            "value"
        ].iloc[0]
        subjects.append(float(value) if pd.notna(value) else np.nan)
    axis.bar(x - 0.19, units, 0.38, color=SKY, label="Samples / merged objects")
    axis.bar(
        x + 0.19,
        np.nan_to_num(subjects),
        0.38,
        color=ORANGE,
        label="Biological subjects",
    )
    for index, value in enumerate(subjects):
        if not np.isfinite(value):
            axis.text(
                index + 0.19,
                0.8,
                "NA",
                ha="center",
                va="bottom",
                color=MID_GRAY,
                fontsize=7,
            )
    axis.set_xticks(x, ["Kang", "AD", "Troph", "Embryo"])
    axis.set_ylabel("Count")
    axis.set_title("Source/preliminary units and subject support")
    axis.legend(loc="upper left")
    _panel_label(axis, "B")

    axis = axes[1, 0]
    data = source[source["panel"] == "C"]
    subset_order = order[:3]
    x = np.arange(len(subset_order))
    mapped = [
        float(
            data[
                (data["dataset"] == item) & (data["metric"] == "gene_mapped_fraction")
            ]["value"].iloc[0]
        )
        for item in subset_order
    ]
    supported = [
        float(
            data[
                (data["dataset"] == item)
                & (data["metric"] == "final_selected_fraction")
            ]["value"].iloc[0]
        )
        for item in subset_order
    ]
    axis.bar(x - 0.19, mapped, 0.38, color=BLUE, label="Gene mapped")
    axis.bar(
        x + 0.19,
        supported,
        0.38,
        color=GREEN,
        label="Final selected after pooled support/top-K",
    )
    axis.set_xticks(x, ["Kang", "AD", "Troph"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Fraction of source interactions")
    axis.set_title("Final resource mapping and top-K selection", pad=28)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=6,
        frameon=False,
    )
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    _panel_label(axis, "C")

    axis = axes[1, 1]
    data = source[source["panel"] == "D"]
    runtime_order = order[:3]
    y = np.arange(len(runtime_order))
    preliminary = np.asarray(
        [
            float(
                data[
                    (data["dataset"] == item)
                    & (data["metric"] == "preliminary_total_compute")
                ]["value"].iloc[0]
            )
            for item in runtime_order
        ]
    )
    final = np.asarray(
        [
            pd.to_numeric(
                data[
                    (data["dataset"] == item)
                    & (data["metric"] == "canonical_final_wall")
                ]["value"].iloc[0],
                errors="coerce",
            )
            for item in runtime_order
        ],
        dtype=float,
    )
    axis.scatter(
        preliminary,
        y,
        color=SKY,
        marker="s",
        s=35,
        label="Preliminary compute",
        zorder=3,
    )
    finite_final = np.isfinite(final)
    axis.scatter(
        final[finite_final],
        y[finite_final],
        color=BLACK,
        marker="D",
        s=34,
        label="Final canonical wall",
        zorder=3,
    )
    for index, (preliminary_value, final_value) in enumerate(
        zip(preliminary, final, strict=True)
    ):
        if np.isfinite(final_value):
            axis.hlines(
                index,
                min(preliminary_value, final_value),
                max(preliminary_value, final_value),
                color=LIGHT_GRAY,
                linewidth=1.2,
                zorder=1,
            )
            axis.text(
                final_value * 1.08,
                index,
                f"{final_value:.1f}s",
                va="center",
                fontsize=6.5,
            )
        else:
            axis.text(
                preliminary_value * 1.15,
                index,
                "final pending",
                va="center",
                color=MID_GRAY,
                fontsize=6,
            )
    axis.set_yticks(y, runtime_order)
    axis.set_xscale("log")
    axis.set_xlabel("Wall-clock seconds (log scale)")
    axis.set_title("Preliminary compute vs final canonical wall")
    axis.legend(loc="upper right")
    _panel_label(axis, "D")

    figure.suptitle(
        "Canonical datasets: scale, resource retention, and runtime",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.10, right=0.97, top=0.90, bottom=0.10, hspace=0.42, wspace=0.34
    )
    _save_figure(figure, figures_dir, "figure01_overview")


def _kang_donor_gene_effects(
    h5ad_path: Path,
    pseudobulk: pd.DataFrame,
    receivers: Sequence[str],
    genes: Sequence[str],
) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path)
    if "counts" not in adata.layers:
        raise ValueError("Kang input is missing the counts layer")
    matrix = adata.layers["counts"]
    counts = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    gene_index = adata.var_names.get_indexer(genes)
    if np.any(gene_index < 0):
        missing = [
            gene for gene, index in zip(genes, gene_index, strict=True) if index < 0
        ]
        raise ValueError(f"Kang input is missing report genes: {missing}")
    obs = adata.obs
    eligible = set(
        map(
            tuple,
            pseudobulk.loc[
                pseudobulk["state_eligible"],
                ["subject_id", "condition", "cell_type"],
            ].itertuples(index=False, name=None),
        )
    )
    rows: list[dict[str, Any]] = []
    subjects = sorted(obs["subject_id"].astype(str).unique())
    for receiver in receivers:
        for subject in subjects:
            context_values: dict[str, np.ndarray] = {}
            for condition in ("ctrl", "stim"):
                key = (subject, condition, receiver)
                if key not in eligible:
                    continue
                mask = (
                    obs["subject_id"].astype(str).eq(subject)
                    & obs["condition"].astype(str).eq(condition)
                    & obs["cell_type"].astype(str).eq(receiver)
                )
                indices = np.flatnonzero(mask.to_numpy())
                library_size = float(counts[indices, :].sum())
                gene_counts = np.asarray(
                    counts[indices, :][:, gene_index].sum(axis=0)
                ).ravel()
                context_values[condition] = np.log1p(
                    gene_counts / library_size * 1_000_000.0
                )
            if set(context_values) != {"ctrl", "stim"}:
                continue
            difference = context_values["stim"] - context_values["ctrl"]
            for gene, value in zip(genes, difference, strict=True):
                rows.append(
                    {
                        "receiver": receiver,
                        "subject_id": subject,
                        "gene": gene,
                        "paired_effect": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _kang_composition(
    pseudobulk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cell_types = sorted(pseudobulk["cell_type"].astype(str).unique())
    wide = pseudobulk.pivot(
        index="subject_id", columns=["condition", "cell_type"], values="cell_proportion"
    ).fillna(0.0)
    donor_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    for subject in wide.index:
        ctrl = np.asarray(
            [
                wide.loc[subject].get(("ctrl", cell_type), 0.0)
                for cell_type in cell_types
            ],
            dtype=float,
        )
        stim = np.asarray(
            [
                wide.loc[subject].get(("stim", cell_type), 0.0)
                for cell_type in cell_types
            ],
            dtype=float,
        )
        changes = np.abs(stim - ctrl)
        donor_rows.append(
            {
                "subject_id": str(subject),
                "spearman_rho": _spearman(ctrl, stim),
                "median_absolute_fraction_change": float(np.median(changes)),
                "max_absolute_fraction_change": float(np.max(changes)),
            }
        )
        for cell_type, ctrl_value, stim_value in zip(
            cell_types, ctrl, stim, strict=True
        ):
            change_rows.append(
                {
                    "subject_id": str(subject),
                    "cell_type": cell_type,
                    "ctrl_fraction": float(ctrl_value),
                    "stim_fraction": float(stim_value),
                    "absolute_fraction_change": float(abs(stim_value - ctrl_value)),
                }
            )
    donors = pd.DataFrame(donor_rows)
    changes = pd.DataFrame(change_rows)
    summary = {
        "paired_subjects": float(len(donors)),
        "median_subject_spearman": float(donors["spearman_rho"].median()),
        "median_absolute_fraction_change": float(
            changes["absolute_fraction_change"].median()
        ),
        "max_absolute_fraction_change": float(
            changes["absolute_fraction_change"].max()
        ),
    }
    return donors, changes, summary


def _paired_availability_edges(
    availability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "sender",
        "receiver",
        "interaction_id",
        "source_interaction_id",
        "ligand",
        "receptor",
        "pathway",
    ]
    table = availability[
        ["subject_id", "condition", *keys, "availability_state"]
    ].copy()
    for column in keys:
        table[column] = table[column].astype("string")
    wide = table.pivot(
        index=["subject_id", *keys], columns="condition", values="availability_state"
    )
    wide = wide.dropna(subset=["ctrl", "stim"])
    wide["paired_effect"] = wide["stim"] - wide["ctrl"]
    donor = wide.reset_index()
    grouped = donor.groupby(keys, sort=False, observed=True)["paired_effect"]
    effects = grouped.agg(
        mean_effect="mean",
        median_effect="median",
        n_pairs="size",
        direction_consistency=lambda values: float(np.mean(values > 0)),
    ).reset_index()
    effects["edge_label"] = (
        effects["sender"]
        + " -> "
        + effects["receiver"]
        + " | "
        + effects["ligand"]
        + "-"
        + effects["receptor"]
    )
    return effects, donor


def _kang_data(
    benchmark_root: Path,
    truth: dict[str, Any],
    book: MetricBook,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_root = benchmark_root / "crychic_kang_prelim"
    response = pd.read_parquet(run_root / "response_contrasts.parquet")
    pseudobulk = pd.read_parquet(run_root / "pseudobulk_units.parquet")
    availability = pd.read_parquet(run_root / "sample_availability.parquet")
    expected = {item["id"]: item for item in truth["expected_observations"]}
    expected_receivers = list(expected["pan_cell_isg_response"]["receiver_cell_types"])

    stim = response[response["contrast"].eq("global:'stim'")].copy()
    isg = stim[
        stim["gene"].isin(ISG_GENES) & stim["receiver"].isin(expected_receivers)
    ].copy()
    donor_receivers = sorted(isg.loc[isg["status"].eq("ok"), "receiver"].unique())
    donor_effects = _kang_donor_gene_effects(
        benchmark_root / "kang2018_batch2.h5ad",
        pseudobulk,
        donor_receivers,
        ISG_GENES,
    )
    consistency = (
        donor_effects.groupby(["receiver", "gene"], observed=True)["paired_effect"]
        .agg(
            donor_direction_consistency=lambda values: float(np.mean(values > 0)),
            donor_pairs="size",
        )
        .reset_index()
    )
    isg = isg.merge(consistency, on=["receiver", "gene"], how="left")
    eligible_isg = isg[isg["status"].eq("ok") & isg["effect"].notna()]
    positive_fraction = float((eligible_isg["effect"] > 0).mean())
    coverage = len(donor_receivers) / len(expected_receivers)
    median_consistency = float(consistency["donor_direction_consistency"].median())

    antigen = stim[stim["gene"].isin(ANTIGEN_GENES) & stim["status"].eq("ok")]
    antigen_positive = float((antigen["effect"] > 0).mean())
    composition_donors, _composition_changes, composition_summary = _kang_composition(
        pseudobulk
    )
    edge_effects, edge_donors = _paired_availability_edges(availability)
    top_edges = (
        edge_effects[edge_effects["n_pairs"].eq(8)].nlargest(9, "mean_effect").copy()
    )
    cxcl10 = edge_effects[
        edge_effects["ligand"].str.contains("CXCL10", case=False, na=False)
        & edge_effects["n_pairs"].eq(8)
    ].nlargest(1, "mean_effect")
    if cxcl10.empty:
        raise ValueError(
            "Kang availability output contains no fully paired CXCL10 edge"
        )
    cx_edge_id = str(cxcl10.iloc[0]["interaction_id"])
    cx_sender = str(cxcl10.iloc[0]["sender"])
    cx_receiver = str(cxcl10.iloc[0]["receiver"])
    cx_donor = edge_donors[
        edge_donors["interaction_id"].eq(cx_edge_id)
        & edge_donors["sender"].eq(cx_sender)
        & edge_donors["receiver"].eq(cx_receiver)
    ].copy()
    ifnb_rows = availability[
        availability["ligand"]
        .astype("string")
        .str.contains("IFNB1", case=False, na=False)
        | availability["source_interaction_id"]
        .astype("string")
        .str.contains("IFNB1", case=False, na=False)
    ]

    final_diagnostics: dict[str, Any] = {
        "status": "not_available",
        "reason_code": "canonical_result_not_complete",
        "ctrl_availability": None,
        "stim_availability": None,
        "availability_delta": None,
        "contrast_records_agree": None,
        "observed_receivers": None,
        "estimable_receivers": None,
        "receiver_coverage": None,
        "positive_ok_isg_rows": None,
        "not_estimable_isg_rows": None,
        "finite_strength_rows": None,
        "positive_strength_rows": None,
        "cxcl10_integrated_strength": None,
        "comm_probability_nonnull_rows": None,
    }
    final_root = benchmark_root / "canonical_v01_results/kang2018"
    final_metrics_path = final_root / "metrics.json"
    final_interactions_path = final_root / "result/interactions.parquet"
    final_responses_path = final_root / "result/responses.parquet"
    if (
        final_metrics_path.is_file()
        and final_interactions_path.is_file()
        and final_responses_path.is_file()
        and _read_json(final_metrics_path).get("status") == "complete"
    ):
        final_interactions = pd.read_parquet(final_interactions_path)
        final_cxcl10 = final_interactions[
            final_interactions["interaction_id"].eq(cx_edge_id)
            & final_interactions["sender"].eq(cx_sender)
            & final_interactions["receiver"].eq(cx_receiver)
            & final_interactions["mode"].eq("state")
        ].copy()
        final_cxcl10["condition"] = final_cxcl10["context_json"].map(
            lambda value: json.loads(str(value)).get("condition")
        )
        context_values = final_cxcl10.groupby("condition", observed=True)[
            "availability"
        ].agg(["mean", "nunique"])
        required_contexts = {"ctrl", "stim"}
        if not required_contexts.issubset(context_values.index):
            raise ValueError(
                "Final Kang CXCL10 diagnostic is missing ctrl or stim retained rows"
            )
        ctrl_availability = float(context_values.loc["ctrl", "mean"])
        stim_availability = float(context_values.loc["stim", "mean"])

        final_responses = pd.read_parquet(final_responses_path)
        final_isg = final_responses[
            final_responses["contrast"].eq("global:'stim'")
            & final_responses["gene"].isin(ISG_GENES)
        ].copy()
        observed_receivers = int(final_isg["receiver"].nunique())
        ok_isg = final_isg[
            final_isg["status"].eq("ok") & final_isg["effect_size"].notna()
        ]
        estimable_receivers = int(ok_isg["receiver"].nunique())
        finite_strength = np.isfinite(
            final_interactions["comm_strength"].to_numpy(dtype=float)
        )
        cxcl10_strength = final_cxcl10["comm_strength"].dropna()
        final_diagnostics = {
            "status": "complete",
            "reason_code": "v0_1_inferential_disabled",
            "ctrl_availability": ctrl_availability,
            "stim_availability": stim_availability,
            "availability_delta": stim_availability - ctrl_availability,
            "contrast_records_agree": bool(
                (context_values.loc[list(required_contexts), "nunique"] == 1).all()
            ),
            "observed_receivers": observed_receivers,
            "estimable_receivers": estimable_receivers,
            "receiver_coverage": (
                estimable_receivers / observed_receivers if observed_receivers else None
            ),
            "positive_ok_isg_rows": int((ok_isg["effect_size"] > 0).sum()),
            "not_estimable_isg_rows": int(
                final_isg["status"].eq("not_estimable").sum()
            ),
            "finite_strength_rows": int(finite_strength.sum()),
            "positive_strength_rows": int(
                (
                    final_interactions.loc[finite_strength, "comm_strength"].to_numpy(
                        dtype=float
                    )
                    > 0
                ).sum()
            ),
            "cxcl10_integrated_strength": (
                float(cxcl10_strength.abs().max())
                if not cxcl10_strength.empty
                else None
            ),
            "comm_probability_nonnull_rows": int(
                final_interactions["comm_probability"].notna().sum()
            ),
        }

    source_parts: list[pd.DataFrame] = []
    isg_source = isg[
        [
            "receiver",
            "gene",
            "effect",
            "standard_error",
            "status",
            "reason_code",
            "donor_direction_consistency",
            "donor_pairs",
        ]
    ].copy()
    isg_source.insert(0, "panel", "A")
    isg_source["data_role"] = "paired_stim_minus_ctrl_log1p_cpm"
    source_parts.append(isg_source)
    comp_source = composition_donors.copy()
    comp_source.insert(0, "panel", "B")
    comp_source["data_role"] = "paired_composition_stability"
    source_parts.append(comp_source)
    top_source = top_edges[
        [
            "sender",
            "receiver",
            "source_interaction_id",
            "ligand",
            "receptor",
            "pathway",
            "mean_effect",
            "median_effect",
            "direction_consistency",
            "n_pairs",
            "edge_label",
        ]
    ].copy()
    top_source.insert(0, "panel", "C")
    top_source["data_role"] = "top_availability_state_edges"
    source_parts.append(top_source)
    cx_source = cx_donor[
        [
            "subject_id",
            "sender",
            "receiver",
            "source_interaction_id",
            "ligand",
            "receptor",
            "pathway",
            "ctrl",
            "stim",
            "paired_effect",
        ]
    ].copy()
    cx_source.insert(0, "panel", "D")
    cx_source["data_role"] = "cxcl10_cxcr3_top_edge_by_subject"
    for name, value in final_diagnostics.items():
        cx_source[f"canonical_final_{name}"] = value
    source_parts.append(cx_source)
    source = pd.concat(source_parts, ignore_index=True, sort=False)

    book.add(
        section="known_biology",
        dataset="Kang IFN-beta",
        metric="paired_subjects_recovered",
        value=composition_summary["paired_subjects"],
        unit="subjects",
        status="pass",
        interpretation="All eight donors have ctrl and stim samples.",
        source_figure="figure02_kang",
    )
    book.add(
        section="known_biology",
        dataset="Kang IFN-beta",
        metric="median_subject_composition_spearman",
        value=composition_summary["median_subject_spearman"],
        unit="rho",
        status=(
            "pass"
            if composition_summary["median_subject_spearman"] >= 0.85
            else "does_not_meet_locked_threshold"
        ),
        interpretation="Locked minimum=0.85.",
        source_figure="figure02_kang",
    )
    book.add(
        section="known_biology",
        dataset="Kang IFN-beta",
        metric="median_absolute_cell_fraction_change",
        value=composition_summary["median_absolute_fraction_change"],
        unit="fraction",
        status=(
            "pass"
            if composition_summary["median_absolute_fraction_change"] <= 0.05
            else "does_not_meet_locked_threshold"
        ),
        interpretation="Locked maximum=0.05.",
        source_figure="figure02_kang",
    )
    book.add(
        section="known_biology",
        dataset="Kang IFN-beta",
        metric="isg_positive_fraction_estimable_pairs",
        value=positive_fraction,
        unit="fraction",
        status=(
            "pass" if positive_fraction >= 0.75 else "does_not_meet_locked_threshold"
        ),
        interpretation=f"{len(eligible_isg)} estimable receiver-gene pairs; locked minimum=0.75.",
        source_figure="figure02_kang",
    )
    book.add(
        section="estimability",
        dataset="Kang IFN-beta",
        metric="isg_receiver_coverage",
        value=coverage,
        unit="fraction",
        status="partial_support" if coverage < 1 else "complete_support",
        reason_code="dendritic_partial_pairing_not_estimable" if coverage < 1 else "",
        interpretation=f"{len(donor_receivers)}/{len(expected_receivers)} locked receiver types were estimable.",
        source_figure="figure02_kang",
    )
    book.add(
        section="known_biology",
        dataset="Kang IFN-beta",
        metric="median_donor_direction_consistency_isg",
        value=median_consistency,
        unit="fraction",
        status=(
            "pass" if median_consistency >= 0.75 else "does_not_meet_locked_threshold"
        ),
        interpretation="Median across estimable receiver-gene pairs; locked minimum=0.75.",
        source_figure="figure02_kang",
    )
    book.add(
        section="known_biology",
        dataset="Kang IFN-beta",
        metric="antigen_presentation_positive_fraction",
        value=antigen_positive,
        unit="fraction",
        status=(
            "pass" if antigen_positive >= 0.67 else "does_not_meet_locked_threshold"
        ),
        interpretation=f"{len(antigen)} estimable receiver-gene pairs; locked minimum=0.67.",
        source_figure="figure02_kang",
    )
    book.add(
        section="availability",
        dataset="Kang IFN-beta",
        metric="top_cxcl10_cxcr3_paired_effect",
        value=float(cxcl10.iloc[0]["mean_effect"]),
        value_text=str(cxcl10.iloc[0]["edge_label"]),
        unit="availability_state_stim_minus_ctrl",
        status="supportive_exploratory",
        resource="CellChatDB human v2",
        interpretation=f"direction_consistency={float(cxcl10.iloc[0]['direction_consistency']):.3f}; n_pairs={int(cxcl10.iloc[0]['n_pairs'])}",
        source_figure="figure02_kang",
    )
    book.add(
        section="interpretation_guardrail",
        dataset="Kang IFN-beta",
        metric="endogenous_ifnb1_supported_rows",
        value=len(ifnb_rows),
        unit="availability_rows",
        status="expected_absence",
        reason_code="exogenous_recombinant_ifn_beta",
        interpretation="Strong receiver ISG response does not require endogenous IFNB1 sender support.",
        source_figure="figure02_kang",
    )
    final_metric_status = (
        "supportive_exploratory"
        if final_diagnostics["status"] == "complete"
        else "not_available"
    )
    final_reason = str(final_diagnostics["reason_code"])
    for metric_name, value, unit, interpretation in (
        (
            "canonical_retained_cxcl10_ctrl_availability",
            final_diagnostics["ctrl_availability"],
            "availability_state",
            "Final retained row after full exploratory branches; not the pure donor-level estimand.",
        ),
        (
            "canonical_retained_cxcl10_stim_availability",
            final_diagnostics["stim_availability"],
            "availability_state",
            "Final retained row after full exploratory branches; not the pure donor-level estimand.",
        ),
        (
            "canonical_retained_cxcl10_availability_delta",
            final_diagnostics["availability_delta"],
            "stim_minus_ctrl_availability_state",
            "Retention diagnostic conditioned on rows entering the full pipeline; preliminary donor-level availability remains the biology estimand.",
        ),
        (
            "canonical_all_observed_isg_receiver_coverage",
            final_diagnostics["receiver_coverage"],
            "fraction",
            f"{final_diagnostics['estimable_receivers']}/{final_diagnostics['observed_receivers']} observed receiver types; Dendritic cells and Megakaryocytes are not estimable.",
        ),
        (
            "canonical_finite_integrated_strength_rows",
            final_diagnostics["finite_strength_rows"],
            "rows",
            f"Positive finite rows={final_diagnostics['positive_strength_rows']}; exploratory in-sample score, not calibrated probability.",
        ),
        (
            "canonical_cxcl10_integrated_strength",
            final_diagnostics["cxcl10_integrated_strength"],
            "absolute_max_strength",
            "CXCL10 retained availability does not imply nonzero integrated attribution/strength.",
        ),
        (
            "canonical_comm_probability_nonnull_rows",
            final_diagnostics["comm_probability_nonnull_rows"],
            "rows",
            "Formal communication probability is disabled in v0.1.",
        ),
    ):
        book.add(
            section="canonical_diagnostic",
            dataset="Kang IFN-beta",
            metric=metric_name,
            value=value,
            unit=unit,
            status=final_metric_status,
            reason_code=final_reason,
            method="CRYCHIC canonical v0.1",
            resource="CellChatDB human v2",
            interpretation=interpretation,
            source_figure="figure02_kang",
        )
    summary = {
        "positive_fraction": positive_fraction,
        "coverage": coverage,
        "median_consistency": median_consistency,
        "antigen_positive": antigen_positive,
        "composition": composition_summary,
        "cxcl10_effect": float(cxcl10.iloc[0]["mean_effect"]),
        "cxcl10_label": str(cxcl10.iloc[0]["edge_label"]),
        "ifnb_rows": len(ifnb_rows),
        "estimable_receivers": donor_receivers,
        "canonical_final": final_diagnostics,
    }
    return source, summary


def _plot_kang(source: pd.DataFrame, figures_dir: Path) -> None:
    figure = plt.figure(figsize=(13.2, 8.2))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.2, 1.2, 0.9],
        height_ratios=[1.0, 1.05],
        hspace=0.46,
        wspace=0.44,
    )
    axis_a = figure.add_subplot(grid[0, :2])
    axis_b = figure.add_subplot(grid[0, 2])
    axis_c = figure.add_subplot(grid[1, :2])
    axis_d = figure.add_subplot(grid[1, 2])

    data = source[source["panel"].eq("A")].copy()
    receiver_order = [
        "B cells",
        "CD14+ Monocytes",
        "CD4 T cells",
        "CD8 T cells",
        "Dendritic cells",
        "FCGR3A+ Monocytes",
        "NK cells",
    ]
    matrix = data.pivot(index="receiver", columns="gene", values="effect").reindex(
        index=receiver_order, columns=ISG_GENES
    )
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values)))
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_orange", [BLUE, "#FFFFFF", VERMILLION]
    )
    cmap.set_bad("#EEEEEE")
    image = axis_a.imshow(
        np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit
    )
    axis_a.set_xticks(np.arange(len(ISG_GENES)), ISG_GENES, rotation=45, ha="right")
    axis_a.set_yticks(np.arange(len(receiver_order)), receiver_order)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "NE" if not np.isfinite(value) else f"{value:.1f}"
            axis_a.text(
                column, row, text, ha="center", va="center", fontsize=5.4, color=BLACK
            )
    colorbar = figure.colorbar(image, ax=axis_a, fraction=0.025, pad=0.02)
    colorbar.set_label("Paired stim - ctrl effect (log1p CPM)")
    axis_a.set_title("Pan-cell interferon-stimulated gene response")
    _panel_label(axis_a, "A")

    data = source[source["panel"].eq("B")].sort_values("subject_id")
    y = np.arange(len(data))
    axis_b.scatter(data["spearman_rho"], y, color=BLUE, s=28, zorder=3)
    axis_b.hlines(y, 0.85, data["spearman_rho"], color=SKY, linewidth=1.0)
    axis_b.axvline(
        0.85, color=VERMILLION, linestyle="--", linewidth=1.0, label="Locked minimum"
    )
    axis_b.set_yticks(y, data["subject_id"])
    axis_b.set_xlim(0.82, 1.01)
    axis_b.set_xlabel("ctrl/stim composition Spearman rho")
    axis_b.set_ylabel("Donor")
    axis_b.set_title("Composition stability")
    axis_b.legend(loc="lower right")
    _panel_label(axis_b, "B")

    data = source[source["panel"].eq("C")].sort_values("mean_effect")
    labels = [
        label.replace(" -> ", " ->\n", 1).replace(" | ", " | ")
        for label in data["edge_label"]
    ]
    colors = [VERMILLION if "CXCL10" in label else BLUE for label in data["edge_label"]]
    bars = axis_c.barh(np.arange(len(data)), data["mean_effect"], color=colors)
    axis_c.set_yticks(np.arange(len(data)), labels)
    axis_c.set_xlabel("Mean paired stim - ctrl availability-state effect")
    axis_c.set_title("Top fully paired molecular-availability edges")
    for bar, consistency in zip(bars, data["direction_consistency"], strict=True):
        axis_c.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{consistency:.0%}",
            va="center",
            fontsize=6,
        )
    _panel_label(axis_c, "C")

    data = source[source["panel"].eq("D")].sort_values("subject_id")
    x = np.arange(len(data))
    axis_d.axhline(0, color=MID_GRAY, linewidth=0.8)
    axis_d.scatter(x, data["paired_effect"], color=VERMILLION, s=32, zorder=3)
    axis_d.hlines(
        data["paired_effect"].mean(),
        -0.35,
        len(data) - 0.65,
        color=BLACK,
        linewidth=1.5,
        label="Mean",
    )
    axis_d.set_xticks(x, data["subject_id"], rotation=45, ha="right")
    axis_d.set_ylabel("Paired availability-state effect")
    axis_d.set_xlabel("Donor")
    edge = data.iloc[0]
    axis_d.set_title(
        f"Pure availability (prelim): CXCL10-CXCR3\n"
        f"{edge['sender']} -> {edge['receiver']}",
        fontsize=7.5,
        pad=7,
    )
    axis_d.legend(loc="upper right")
    final_delta = pd.to_numeric(
        edge.get("canonical_final_availability_delta"), errors="coerce"
    )
    if np.isfinite(final_delta):
        axis_d.text(
            0.02,
            0.96,
            f"Prelim donor mean = {data['paired_effect'].mean():.3f}\n"
            f"Final-retained diagnostic = {final_delta:.3f}",
            transform=axis_d.transAxes,
            fontsize=5.8,
            va="top",
        )
    axis_d.text(
        -0.20,
        1.10,
        "D",
        transform=axis_d.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    figure.suptitle(
        "Kang 2018: paired IFN-beta biology and availability",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.09)
    _save_figure(figure, figures_dir, "figure02_kang")


def _ad_liana_data(
    benchmark_root: Path,
    database_root: Path,
    repo_root: Path,
    book: MetricBook,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_root = benchmark_root / "cellchat_reference"
    compositions = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(reference_root.glob("ad_*/cell_composition.csv"))
        ],
        ignore_index=True,
    )
    pathway_summary = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(reference_root.glob("ad_*/pathway_summary.csv"))
        ],
        ignore_index=True,
    )
    pathway_edges = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(reference_root.glob("ad_*/pathway_edges.csv"))
        ],
        ignore_index=True,
    )
    availability = pd.read_parquet(
        benchmark_root / "crychic_ad_prelim/sample_availability.parquet"
    )
    for column in ("pathway", "sender", "receiver"):
        availability[column] = availability[column].astype("string")
    canonical_metrics_path = (
        benchmark_root / "canonical_v01_results/ad_skin/metrics.json"
    )
    canonical_interactions_path = (
        benchmark_root / "canonical_v01_results/ad_skin/result/interactions.parquet"
    )
    canonical_ready = False
    if canonical_metrics_path.is_file() and canonical_interactions_path.is_file():
        canonical_metrics = _read_json(canonical_metrics_path)
        canonical_ready = canonical_metrics.get("status") == "complete"

    if canonical_ready:
        bundle = load_cellchat_resource(
            database_root,
            "human",
            manifest_path=repo_root / "resources/cellchatdb_human_v2.json",
        )
        pathway_map = pd.DataFrame(
            {
                "interaction_id": [item.interaction_id for item in bundle.interactions],
                "pathway": [item.pathway for item in bundle.interactions],
            }
        ).drop_duplicates("interaction_id")
        corrected = pd.read_parquet(canonical_interactions_path)
        corrected = corrected[
            corrected["mode"].eq("state")
            & corrected["contrast"].eq("availability_only")
            & corrected["availability"].notna()
        ].copy()
        corrected["context"] = corrected["context_json"].map(
            lambda value: str(json.loads(value)["context"])
        )
        corrected = corrected.merge(
            pathway_map,
            on="interaction_id",
            how="left",
            validate="many_to_one",
        )
        if corrected["pathway"].isna().any():
            missing = int(corrected["pathway"].isna().sum())
            raise ValueError(
                f"corrected AD interactions contain {missing} unmapped pathway rows"
            )
        cry_pathways = (
            corrected.groupby(["context", "pathway"], observed=True)["availability"]
            .sum(min_count=1)
            .rename("availability_state")
            .reset_index()
        )
        cry_edges = (
            corrected.groupby(
                ["context", "pathway", "sender", "receiver"], observed=True
            )["availability"]
            .sum(min_count=1)
            .rename("availability_state")
            .reset_index()
        )
        crychic_output_source = "corrected_canonical_v01_interactions"
        crychic_metric_status = "supportive_exploratory_corrected"
        crychic_method = "CRYCHIC canonical v0.1 corrected availability"
    else:
        per_sample_pathway = (
            availability.groupby(["context", "sample_id", "pathway"], observed=True)[
                "availability_state"
            ]
            .sum(min_count=1)
            .reset_index()
        )
        cry_pathways = (
            per_sample_pathway.groupby(["context", "pathway"], observed=True)[
                "availability_state"
            ]
            .mean()
            .reset_index()
        )
        cry_edges = (
            availability.groupby(
                ["context", "sample_id", "pathway", "sender", "receiver"],
                observed=True,
            )["availability_state"]
            .sum(min_count=1)
            .reset_index()
            .groupby(["context", "pathway", "sender", "receiver"], observed=True)[
                "availability_state"
            ]
            .mean()
            .reset_index()
        )
        crychic_output_source = "preliminary_normalized_only_output"
        crychic_metric_status = "invalidated_pending_corrected_rerun"
        crychic_method = "CRYCHIC preliminary availability"

    inflammatory_types = ["Inflam. FIB", "Inflam. DC", "Inflam. TC"]
    inflammation = compositions[
        compositions["cell_type"].isin(inflammatory_types)
    ].copy()
    comp_wide = inflammation.pivot(
        index="cell_type", columns="context", values="cell_fraction"
    )

    shared_path_rows: list[pd.DataFrame] = []
    mif_edge_rows: list[pd.DataFrame] = []
    pathway_stats: dict[str, dict[str, float]] = {}
    for context in ("NL", "LS"):
        cellchat = pathway_summary[pathway_summary["context"].eq(context)].copy()
        cry = cry_pathways[cry_pathways["context"].eq(context)].copy()
        cellchat["cellchat_global_rank"] = cellchat["total_strength"].rank(
            method="min", ascending=False
        )
        cry["crychic_global_rank"] = cry["availability_state"].rank(
            method="min", ascending=False
        )
        shared = cellchat.merge(cry, on=["context", "pathway"], validate="one_to_one")
        shared["cellchat_shared_rank"] = shared["total_strength"].rank(
            method="min", ascending=False
        )
        shared["crychic_shared_rank"] = shared["availability_state"].rank(
            method="min", ascending=False
        )
        shared["rank_spearman"] = _spearman(
            shared["cellchat_shared_rank"], shared["crychic_shared_rank"]
        )
        shared_path_rows.append(shared)

        cellchat_mif = pathway_edges[
            pathway_edges["context"].eq(context) & pathway_edges["pathway"].eq("MIF")
        ]
        cry_mif = cry_edges[
            cry_edges["context"].eq(context) & cry_edges["pathway"].eq("MIF")
        ]
        merged = cellchat_mif.merge(
            cry_mif,
            on=["context", "pathway", "sender", "receiver"],
            validate="one_to_one",
        )
        rho = _spearman(merged["strength"], merged["availability_state"])
        merged["edge_rank_spearman"] = rho
        mif_edge_rows.append(merged)
        mif_row = shared[shared["pathway"].eq("MIF")].iloc[0]
        pathway_stats[context] = {
            "rank_spearman": float(shared["rank_spearman"].iloc[0]),
            "mif_edge_spearman": rho,
            "mif_shared_edges": float(len(merged)),
            "cellchat_mif_rank": float(mif_row["cellchat_global_rank"]),
            "crychic_mif_rank": float(mif_row["crychic_global_rank"]),
            "cellchat_mif_strength": float(mif_row["total_strength"]),
            "crychic_mif_availability": float(mif_row["availability_state"]),
        }

    liana = pd.read_parquet(
        benchmark_root / "liana_kang/liana_by_sample.parquet",
        columns=[
            "sample_id",
            "source",
            "target",
            "ligand_complex",
            "receptor_complex",
            "magnitude_rank",
            "subject_id",
            "condition",
        ],
    )
    edge_keys = ["source", "target", "ligand_complex", "receptor_complex"]
    liana_rows: list[dict[str, Any]] = []
    for subject, group in liana.groupby("subject_id", observed=True):
        ctrl = group[group["condition"].eq("ctrl")][[*edge_keys, "magnitude_rank"]]
        stim = group[group["condition"].eq("stim")][[*edge_keys, "magnitude_rank"]]
        shared = ctrl.merge(
            stim, on=edge_keys, suffixes=("_ctrl", "_stim"), validate="one_to_one"
        )
        top_ctrl = set(
            map(
                tuple,
                ctrl.nsmallest(500, "magnitude_rank")[edge_keys].itertuples(
                    index=False, name=None
                ),
            )
        )
        top_stim = set(
            map(
                tuple,
                stim.nsmallest(500, "magnitude_rank")[edge_keys].itertuples(
                    index=False, name=None
                ),
            )
        )
        union = top_ctrl | top_stim
        liana_rows.append(
            {
                "subject_id": str(subject),
                "shared_edges": len(shared),
                "magnitude_rank_spearman": _spearman(
                    shared["magnitude_rank_ctrl"], shared["magnitude_rank_stim"]
                ),
                "top500_jaccard": len(top_ctrl & top_stim) / len(union),
            }
        )
    liana_summary = pd.DataFrame(liana_rows)

    source_parts: list[pd.DataFrame] = []
    panel_a = inflammation.copy()
    panel_a.insert(0, "panel", "A")
    panel_a["data_role"] = "ad_inflammatory_cell_composition"
    panel_a["output_source"] = "archived_cellchat_reference_only"
    source_parts.append(panel_a)
    panel_b = pd.concat(mif_edge_rows, ignore_index=True)
    panel_b.insert(0, "panel", "B")
    panel_b["data_role"] = "mif_sender_receiver_concordance"
    panel_b["output_source"] = crychic_output_source
    panel_b["metric_status"] = crychic_metric_status
    source_parts.append(panel_b)
    panel_c = pd.concat(shared_path_rows, ignore_index=True)
    panel_c.insert(0, "panel", "C")
    panel_c["data_role"] = "shared_pathway_rank_concordance"
    panel_c["output_source"] = crychic_output_source
    panel_c["metric_status"] = crychic_metric_status
    source_parts.append(panel_c)
    panel_d = liana_summary.copy()
    panel_d.insert(0, "panel", "D")
    panel_d["data_role"] = "liana_default_resource_within_donor_rank_stability"
    panel_d["output_source"] = "liana_consensus_default"
    source_parts.append(panel_d)
    source = pd.concat(source_parts, ignore_index=True, sort=False)

    for cell_type in inflammatory_types:
        nl = float(comp_wide.loc[cell_type, "NL"])
        ls = float(comp_wide.loc[cell_type, "LS"])
        book.add(
            section="known_biology",
            dataset="AD skin",
            metric=f"{cell_type}_LS_minus_NL_fraction",
            value=ls - nl,
            unit="cell_fraction",
            status="supportive_exploratory" if ls > nl else "direction_not_recovered",
            method="CellChat archived reference",
            resource="CellChat archived AD objects",
            interpretation=f"NL={nl:.6f}; LS={ls:.6f}; unpaired normalized-only data.",
            source_figure="figure03_ad_liana",
        )
    for context, values in pathway_stats.items():
        book.add(
            section="reference_reproduction",
            dataset="AD skin",
            metric=f"{context}_shared_pathway_rank_spearman",
            value=values["rank_spearman"],
            unit="rho",
            status=crychic_metric_status,
            reason_code=(
                "normalized_only_detection_shrink_bug_in_preliminary_output"
                if not canonical_ready
                else ""
            ),
            method=crychic_method,
            resource="CellChatDB human v2",
            interpretation="Rank comparison on pathways shared with the archived CellChat summary.",
            source_figure="figure03_ad_liana",
        )
        book.add(
            section="reference_reproduction",
            dataset="AD skin",
            metric=f"{context}_MIF_sender_receiver_rank_spearman",
            value=values["mif_edge_spearman"],
            unit="rho",
            status=crychic_metric_status,
            reason_code=(
                "normalized_only_detection_shrink_bug_in_preliminary_output"
                if not canonical_ready
                else ""
            ),
            method=crychic_method,
            resource="CellChatDB human v2",
            interpretation=f"shared_edges={int(values['mif_shared_edges'])}; CellChat score is not treated as probability truth.",
            source_figure="figure03_ad_liana",
        )
        book.add(
            section="reference_reproduction",
            dataset="AD skin",
            metric=f"{context}_MIF_CRYCHIC_global_rank",
            value=values["crychic_mif_rank"],
            unit="rank",
            status=crychic_metric_status,
            reason_code=(
                "normalized_only_detection_shrink_bug_in_preliminary_output"
                if not canonical_ready
                else ""
            ),
            method=crychic_method,
            resource="CellChatDB human v2",
            interpretation=f"Archived CellChat MIF global rank={int(values['cellchat_mif_rank'])}.",
            source_figure="figure03_ad_liana",
        )
    book.add(
        section="method_stability",
        dataset="Kang IFN-beta",
        metric="LIANA_default_median_ctrl_stim_rank_spearman",
        value=float(liana_summary["magnitude_rank_spearman"].median()),
        unit="rho",
        status="descriptive_rank_only",
        method="LIANA+ rank_aggregate.by_sample 1.7.3",
        resource="LIANA consensus default",
        reason_code="resource_not_harmonized_with_crychic",
        interpretation="Within-method, within-donor rank stability; not differential calibration or cross-method accuracy.",
        source_figure="figure03_ad_liana",
    )
    book.add(
        section="method_stability",
        dataset="Kang IFN-beta",
        metric="LIANA_default_median_ctrl_stim_top500_jaccard",
        value=float(liana_summary["top500_jaccard"].median()),
        unit="jaccard",
        status="descriptive_rank_only",
        method="LIANA+ rank_aggregate.by_sample 1.7.3",
        resource="LIANA consensus default",
        reason_code="resource_not_harmonized_with_crychic",
        interpretation="Top-500 overlap on each donor's two condition-specific default-resource results.",
        source_figure="figure03_ad_liana",
    )
    summary = {
        "inflammatory": {
            cell_type: {
                "NL": float(comp_wide.loc[cell_type, "NL"]),
                "LS": float(comp_wide.loc[cell_type, "LS"]),
            }
            for cell_type in inflammatory_types
        },
        "pathway_stats": pathway_stats,
        "crychic_output_source": crychic_output_source,
        "crychic_metric_status": crychic_metric_status,
        "liana_median_rho": float(liana_summary["magnitude_rank_spearman"].median()),
        "liana_median_jaccard": float(liana_summary["top500_jaccard"].median()),
    }
    return source, summary


def _plot_ad_liana(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))

    axis = axes[0, 0]
    data = source[source["panel"].eq("A")]
    types = ["Inflam. FIB", "Inflam. DC", "Inflam. TC"]
    x = np.arange(len(types))
    for offset, context, color in ((-0.19, "NL", BLUE), (0.19, "LS", VERMILLION)):
        values = [
            float(
                data[(data["cell_type"].eq(cell_type)) & (data["context"].eq(context))][
                    "cell_fraction"
                ].iloc[0]
            )
            for cell_type in types
        ]
        axis.bar(x + offset, values, 0.38, color=color, label=context)
    axis.set_xticks(x, types, rotation=20, ha="right")
    axis.set_ylabel("Cell fraction")
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis.set_title("Archived CellChat composition (reference-only)")
    axis.legend(loc="upper right")
    _panel_label(axis, "A")

    axis = axes[0, 1]
    data = source[source["panel"].eq("B")]
    for context, color, marker in (("NL", BLUE, "o"), ("LS", VERMILLION, "s")):
        subset = data[data["context"].eq(context)]
        axis.scatter(
            subset["strength"],
            subset["availability_state"],
            color=color,
            marker=marker,
            s=18,
            alpha=0.72,
            label=f"{context}: rho={subset['edge_rank_spearman'].iloc[0]:.2f}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Archived CellChat MIF edge strength")
    axis.set_ylabel("CRYCHIC MIF availability state")
    axis.set_title("MIF sender-receiver rank reproduction")
    axis.legend(loc="lower right")
    _panel_label(axis, "B")

    axis = axes[1, 0]
    data = source[source["panel"].eq("C")]
    for context, color, marker in (("NL", BLUE, "o"), ("LS", VERMILLION, "s")):
        subset = data[data["context"].eq(context)]
        axis.scatter(
            subset["cellchat_shared_rank"],
            subset["crychic_shared_rank"],
            color=color,
            marker=marker,
            s=24,
            alpha=0.75,
            label=f"{context}: rho={subset['rank_spearman'].iloc[0]:.2f}",
        )
        mif = subset[subset["pathway"].eq("MIF")].iloc[0]
        offset = (-52, -9) if context == "NL" else (8, 9)
        axis.annotate(
            f"MIF ({context})",
            (mif["cellchat_shared_rank"], mif["crychic_shared_rank"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
            color=color,
        )
    max_rank = int(
        max(data["cellchat_shared_rank"].max(), data["crychic_shared_rank"].max())
    )
    axis.plot([1, max_rank], [1, max_rank], color=LIGHT_GRAY, linestyle="--", zorder=0)
    axis.invert_xaxis()
    axis.invert_yaxis()
    axis.set_xlabel("CellChat rank among shared pathways (1 = top)")
    axis.set_ylabel("CRYCHIC rank among shared pathways (1 = top)")
    axis.set_title("Pathway-order concordance")
    axis.legend(loc="lower right")
    _panel_label(axis, "C")

    axis = axes[1, 1]
    data = source[source["panel"].eq("D")].sort_values("subject_id")
    x = np.arange(len(data))
    axis.plot(
        x,
        data["magnitude_rank_spearman"],
        color=BLUE,
        marker="o",
        label="Shared-edge Spearman rho",
    )
    axis.plot(
        x, data["top500_jaccard"], color=ORANGE, marker="s", label="Top-500 Jaccard"
    )
    axis.set_xticks(x, data["subject_id"], rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Within-donor consistency")
    axis.set_xlabel("Donor")
    axis.set_title("LIANA default-resource rank stability")
    axis.legend(loc="upper left")
    _panel_label(axis, "D")

    figure.suptitle(
        "AD archived reference vs CRYCHIC and LIANA rank-only comparator",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.10, right=0.97, top=0.91, bottom=0.11, hspace=0.42, wspace=0.32
    )
    _save_figure(figure, figures_dir, "figure03_ad_liana")


def _troph_embryo_data(
    benchmark_root: Path,
    database_root: Path,
    book: MetricBook,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    troph_root = benchmark_root / "crychic_trophoblast_prelim"
    response = pd.read_parquet(troph_root / "response_context_means.parquet")
    availability = pd.read_parquet(troph_root / "sample_availability.parquet")
    active_tf = pd.read_csv(
        benchmark_root / "trophoblast/data/active_TFs.tsv", sep="\t"
    )
    deg = pd.read_csv(
        benchmark_root / "trophoblast/data/DEGs_inv_trophoblast.tsv",
        sep="\t",
        usecols=["cluster", "gene", "logFC"],
    )
    genes = pd.read_parquet(
        database_root / "cellphonedb/v5.0.0/genes.parquet",
        columns=["hgnc_symbol", "protein_multidata_id", "name"],
    )
    hla_entities = genes.loc[
        genes["hgnc_symbol"].eq("HLA-G"), ["protein_multidata_id", "name"]
    ].drop_duplicates()
    hla_ids = set(hla_entities["protein_multidata_id"].astype(int))
    interactions = pd.read_parquet(
        database_root / "cellphonedb/v5.0.0/interactions.parquet",
        columns=[
            "id_cp_interaction",
            "multidata_1_id",
            "multidata_2_id",
            "directionality",
            "name_1",
            "name_2",
        ],
    )
    hla_interactions = interactions[
        interactions["multidata_1_id"].astype(int).isin(hla_ids)
        | interactions["multidata_2_id"].astype(int).isin(hla_ids)
    ]
    hla_source_ids = set(hla_interactions["id_cp_interaction"].astype(str))
    hla_display_names = set(hla_entities["name"].astype(str))

    hla_response = response[response["gene"].eq("HLA-G")].copy()
    ok_response = response[response["status"].eq("ok")].copy()
    ok_response["gene_rank"] = ok_response.groupby("receiver", observed=True)[
        "mean_response"
    ].rank(method="min", ascending=False)
    ok_response["top_fraction"] = ok_response.groupby("receiver", observed=True)[
        "mean_response"
    ].rank(method="average", pct=True, ascending=False)
    hla_response = hla_response.merge(
        ok_response[["receiver", "gene", "gene_rank", "top_fraction"]],
        on=["receiver", "gene"],
        how="left",
    )
    hla_deg = deg[deg["gene"].eq("HLA-G")][["cluster", "logFC"]].rename(
        columns={"cluster": "receiver", "logFC": "external_logFC"}
    )
    hla_response = hla_response.merge(hla_deg, on="receiver", how="left")

    availability["source_interaction_id"] = availability[
        "source_interaction_id"
    ].astype("string")
    availability["sender"] = availability["sender"].astype("string")
    availability["ligand"] = availability["ligand"].astype("string")
    availability["interaction_id"] = availability["interaction_id"].astype("string")
    ligand_scores = (
        availability.groupby(["sender", "ligand"], observed=True)["ligand_availability"]
        .mean()
        .reset_index()
    )
    ligand_scores["ligand_rank"] = ligand_scores.groupby("sender", observed=True)[
        "ligand_availability"
    ].rank(method="min", ascending=False)
    ligand_scores["top_fraction"] = ligand_scores.groupby("sender", observed=True)[
        "ligand_availability"
    ].rank(method="average", pct=True, ascending=False)
    hla_ligand = ligand_scores[ligand_scores["ligand"].isin(hla_display_names)].copy()
    hla_ligand["gene"] = "HLA-G"
    hla_ligand["source_interactions"] = ";".join(sorted(hla_source_ids))

    hla_mapping = availability[
        availability["source_interaction_id"].isin(hla_source_ids)
    ][
        [
            "interaction_id",
            "source_interaction_id",
            "ligand",
            "receptor",
        ]
    ].drop_duplicates("interaction_id")
    hla_final = {
        "available": False,
        "resource_interactions": len(hla_source_ids),
        "retained_interactions": None,
        "max_complete_edge_availability": None,
        "max_complete_edge_label": "not_available",
        "evt1_self_availability": None,
    }
    hla_final_rows = pd.DataFrame()
    final_metrics_path = (
        benchmark_root / "canonical_v01_results/trophoblast/metrics.json"
    )
    final_interactions_path = (
        benchmark_root / "canonical_v01_results/trophoblast/result/interactions.parquet"
    )
    if final_metrics_path.is_file() and final_interactions_path.is_file():
        final_metrics = _read_json(final_metrics_path)
        if final_metrics.get("status") == "complete":
            final = pd.read_parquet(final_interactions_path)
            hla_final_rows = final[
                final["interaction_id"]
                .astype("string")
                .isin(set(hla_mapping["interaction_id"]))
                & final["mode"].eq("state")
                & final["availability"].notna()
            ].copy()
            hla_final_rows = hla_final_rows.merge(
                hla_mapping,
                on="interaction_id",
                how="left",
                validate="many_to_one",
            )
            retained = int(hla_final_rows["interaction_id"].nunique())
            if hla_final_rows.empty:
                maximum = 0.0
                maximum_label = "none"
                evt1_self = 0.0
            else:
                maximum_row = hla_final_rows.loc[
                    hla_final_rows["availability"].idxmax()
                ]
                maximum = float(maximum_row["availability"])
                maximum_label = (
                    f"{maximum_row['sender']} -> {maximum_row['receiver']} | "
                    f"{maximum_row['source_interaction_id']}"
                )
                evt1 = hla_final_rows[
                    hla_final_rows["sender"].eq("EVT_1")
                    & hla_final_rows["receiver"].eq("EVT_1")
                ]
                evt1_self = float(evt1["availability"].max()) if not evt1.empty else 0.0
            hla_final = {
                "available": True,
                "resource_interactions": len(hla_source_ids),
                "retained_interactions": retained,
                "max_complete_edge_availability": maximum,
                "max_complete_edge_label": maximum_label,
                "evt1_self_availability": evt1_self,
            }
            book.add(
                section="canonical_complete_edge",
                dataset="Trophoblast",
                metric="HLA-G_resource_interactions_retained",
                value=retained,
                value_text=f"{retained}/{len(hla_source_ids)}",
                unit="interactions",
                status="complete_edge_limited",
                method="CRYCHIC canonical v0.1",
                resource="CellPhoneDB human v5.0.0",
                interpretation="Top-800 resource truncation; HLA-G ligand support is reported separately.",
                source_figure="figure04_trophoblast_embryo",
            )
            book.add(
                section="canonical_complete_edge",
                dataset="Trophoblast",
                metric="HLA-G_max_complete_edge_availability",
                value=maximum,
                value_text=maximum_label,
                unit="state_availability",
                status="complete_edge_limited",
                method="CRYCHIC canonical v0.1",
                resource="CellPhoneDB human v5.0.0",
                interpretation=f"EVT_1->EVT_1 availability={evt1_self:.6f}; no strong EVT complete edge.",
                source_figure="figure04_trophoblast_embryo",
            )

    tf_clusters = ["EVT_1", "EVT_2", "iEVT"]
    tf_expression = response[
        response["receiver"].isin(tf_clusters) & response["gene"].isin(TF_GENES)
    ][
        ["receiver", "gene", "mean_response", "n_subjects", "status", "reason_code"]
    ].copy()
    tf_active = (
        active_tf[
            active_tf["cluster"].isin(tf_clusters) & active_tf["TF"].isin(TF_GENES)
        ]
        .assign(external_active_tf=True)
        .rename(columns={"cluster": "receiver", "TF": "gene"})
    )
    tf_grid = pd.MultiIndex.from_product(
        [tf_clusters, TF_GENES], names=["receiver", "gene"]
    ).to_frame(index=False)
    tf_grid = tf_grid.merge(tf_expression, on=["receiver", "gene"], how="left")
    tf_grid = tf_grid.merge(tf_active, on=["receiver", "gene"], how="left")
    tf_grid["external_active_tf"] = tf_grid["external_active_tf"].fillna(False)

    reference_root = benchmark_root / "cellchat_reference"
    embryo_composition = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(reference_root.glob("embryo_*/cell_composition.csv"))
        ],
        ignore_index=True,
    )
    embryo_pathways = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(reference_root.glob("embryo_*/pathway_summary.csv"))
        ],
        ignore_index=True,
    )
    embryo_edges = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(reference_root.glob("embryo_*/pathway_edges.csv"))
        ],
        ignore_index=True,
    )
    selected_cell_types = [
        "FIB-A",
        "FIB-B",
        "FIB-P",
        "Basal",
        "Spinious",
        "DC",
        "Pericyte",
    ]
    comp_grid = pd.MultiIndex.from_product(
        [["E13.5", "E14.5"], selected_cell_types], names=["context", "cell_type"]
    ).to_frame(index=False)
    embryo_selected = comp_grid.merge(
        embryo_composition,
        on=["context", "cell_type"],
        how="left",
    )
    embryo_selected["status"] = np.where(
        embryo_selected["n_cells"].isna(), "structural_absence", "observed"
    )
    wnt = embryo_pathways[embryo_pathways["pathway"].isin(["WNT", "ncWNT"])].copy()
    wnt["global_rank"] = wnt.groupby("context", observed=True)["total_strength"].rank(
        method="min", ascending=False
    )
    jaccard_rows: list[dict[str, Any]] = []
    for pathway in ("WNT", "ncWNT"):
        stage_sets: dict[str, set[tuple[str, str]]] = {}
        for context in ("E13.5", "E14.5"):
            top = embryo_edges[
                embryo_edges["context"].eq(context)
                & embryo_edges["pathway"].eq(pathway)
            ].nlargest(20, "strength")
            stage_sets[context] = set(
                map(
                    tuple,
                    top[["sender", "receiver"]].itertuples(index=False, name=None),
                )
            )
        union = stage_sets["E13.5"] | stage_sets["E14.5"]
        jaccard_rows.append(
            {
                "pathway": pathway,
                "top20_sender_receiver_jaccard": len(
                    stage_sets["E13.5"] & stage_sets["E14.5"]
                )
                / len(union)
                if union
                else np.nan,
            }
        )
    jaccard = pd.DataFrame(jaccard_rows)
    wnt = wnt.merge(jaccard, on="pathway", how="left")

    source_parts: list[pd.DataFrame] = []
    panel_a = hla_response.copy()
    panel_a.insert(0, "panel", "A")
    panel_a["data_role"] = "hla_g_receiver_expression"
    source_parts.append(panel_a)
    panel_b = hla_ligand.copy()
    panel_b.insert(0, "panel", "B")
    panel_b["data_role"] = "hla_g_ligand_availability_rank"
    source_parts.append(panel_b)
    if not hla_final_rows.empty:
        panel_b_final = hla_final_rows.copy()
        panel_b_final.insert(0, "panel", "B_final")
        panel_b_final["data_role"] = "hla_g_final_complete_edge_availability"
        source_parts.append(panel_b_final)
    panel_c = tf_grid.copy()
    panel_c.insert(0, "panel", "C")
    panel_c["data_role"] = "tutorial_active_tf_support"
    source_parts.append(panel_c)
    panel_d = embryo_selected.copy()
    panel_d.insert(0, "panel", "D")
    panel_d["data_role"] = "embryonic_stage_composition"
    source_parts.append(panel_d)
    panel_e = wnt.copy()
    panel_e.insert(0, "panel", "E")
    panel_e["data_role"] = "embryonic_wnt_reference"
    source_parts.append(panel_e)
    source = pd.concat(source_parts, ignore_index=True, sort=False)

    for receiver in ("EVT_1", "iEVT", "eEVT"):
        row = hla_response[hla_response["receiver"].eq(receiver)]
        if row.empty:
            continue
        book.add(
            section="known_biology",
            dataset="Trophoblast",
            metric=f"HLA-G_expression_{receiver}",
            value=float(row.iloc[0]["mean_response"]),
            unit="mean_log_normalized_expression",
            status="supportive_exploratory",
            resource="CellPhoneDB tutorial expression",
            interpretation=f"within-receiver gene rank={int(row.iloc[0]['gene_rank'])}; top_fraction={float(row.iloc[0]['top_fraction']):.4f}",
            source_figure="figure04_trophoblast_embryo",
        )
        ligand = hla_ligand[hla_ligand["sender"].eq(receiver)]
        if not ligand.empty:
            book.add(
                section="known_biology",
                dataset="Trophoblast",
                metric=f"HLA-G_ligand_availability_{receiver}",
                value=float(ligand.iloc[0]["ligand_availability"]),
                unit="ligand_availability",
                status="supportive_exploratory",
                resource="CellPhoneDB human v5.0.0",
                interpretation=f"within-sender ligand rank={int(ligand.iloc[0]['ligand_rank'])}; top_fraction={float(ligand.iloc[0]['top_fraction']):.4f}",
                source_figure="figure04_trophoblast_embryo",
            )
    expected_tf_pairs = len(tf_clusters) * len(TF_GENES)
    external_tf_pairs = int(tf_grid["external_active_tf"].sum())
    book.add(
        section="known_biology",
        dataset="Trophoblast",
        metric="expected_TF_external_active_pair_fraction",
        value=external_tf_pairs / expected_tf_pairs,
        unit="fraction",
        status="supportive_silver_standard",
        resource="CellPhoneDB tutorial active_TFs.tsv",
        interpretation=f"{external_tf_pairs}/{expected_tf_pairs} cluster-TF pairs; external labels are supportive, not independent truth.",
        source_figure="figure04_trophoblast_embryo",
    )
    for cell_type in ("DC", "Pericyte"):
        row = embryo_selected[
            embryo_selected["context"].eq("E13.5")
            & embryo_selected["cell_type"].eq(cell_type)
        ].iloc[0]
        book.add(
            section="structural_absence",
            dataset="Embryonic skin",
            metric=f"E13.5_{cell_type}_state",
            value=None,
            value_text="structural_absence",
            unit="not_a_zero",
            status=str(row["status"]),
            reason_code="stage_specific_cell_type",
            method="CellChat archived reference",
            resource="CellChat archived embryonic-skin objects",
            interpretation="Missing stage/cell-type state is not measured zero.",
            source_figure="figure04_trophoblast_embryo",
        )
    for pathway in ("WNT", "ncWNT"):
        subset = wnt[wnt["pathway"].eq(pathway)].set_index("context")
        fold = float(
            subset.loc["E14.5", "total_strength"]
            / subset.loc["E13.5", "total_strength"]
        )
        edge_fold = float(
            subset.loc["E14.5", "nonzero_edges"] / subset.loc["E13.5", "nonzero_edges"]
        )
        jac = float(subset["top20_sender_receiver_jaccard"].iloc[0])
        book.add(
            section="known_biology",
            dataset="Embryonic skin",
            metric=f"{pathway}_E14_to_E13_total_strength_ratio",
            value=fold,
            unit="ratio",
            status="supportive_reconfiguration",
            method="CellChat archived reference",
            resource="CellChat archived embryonic-skin objects",
            interpretation=f"nonzero-edge ratio={edge_fold:.3f}; top-20 sender-receiver Jaccard={jac:.3f}; no subject replicates.",
            source_figure="figure04_trophoblast_embryo",
        )
    summary = {
        "hla_response": hla_response[
            hla_response["receiver"].isin(["EVT_1", "iEVT", "eEVT"])
        ][["receiver", "mean_response", "gene_rank", "top_fraction"]].to_dict(
            "records"
        ),
        "hla_ligand": hla_ligand[hla_ligand["sender"].isin(["EVT_1", "iEVT", "eEVT"])][
            ["sender", "ligand_availability", "ligand_rank", "top_fraction"]
        ].to_dict("records"),
        "tf_external_pairs": external_tf_pairs,
        "tf_expected_pairs": expected_tf_pairs,
        "hla_final_complete_edge": hla_final,
        "wnt": wnt.to_dict("records"),
    }
    return source, summary


def _plot_troph_embryo(source: pd.DataFrame, figures_dir: Path) -> None:
    figure = plt.figure(figsize=(13.2, 8.4))
    grid = figure.add_gridspec(2, 3, hspace=0.48, wspace=0.38)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[0, 2])
    axis_d = figure.add_subplot(grid[1, :2])
    axis_e = figure.add_subplot(grid[1, 2])

    data = source[source["panel"].eq("A")].copy()
    order = ["EVT_1", "EVT_2", "iEVT", "eEVT", "SCT", "VCT", "VCT_p"]
    data = data.set_index("receiver").reindex(order).reset_index()
    values = data["mean_response"].to_numpy(dtype=float)
    colors = [
        VERMILLION if receiver in {"EVT_1", "iEVT", "eEVT"} else SKY
        for receiver in order
    ]
    bars = axis_a.bar(np.arange(len(order)), np.nan_to_num(values), color=colors)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            axis_a.text(
                index,
                0.12,
                "NE",
                rotation=90,
                ha="center",
                va="bottom",
                color=MID_GRAY,
                fontsize=6,
            )
        else:
            rank = data.iloc[index]["gene_rank"]
            axis_a.text(
                index,
                value + 0.12,
                f"r{int(rank)}",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )
    axis_a.set_xticks(np.arange(len(order)), order, rotation=45, ha="right")
    axis_a.set_ylabel("Mean log-normalized expression")
    axis_a.set_title("HLA-G receiver expression")
    _panel_label(axis_a, "A")

    data = source[
        source["panel"].eq("B") & source["sender"].isin(["EVT_1", "iEVT", "eEVT", "GC"])
    ].copy()
    data = data.sort_values("ligand_availability")
    bars = axis_b.barh(data["sender"], data["ligand_availability"], color=GREEN)
    for bar, rank, fraction in zip(
        bars, data["ligand_rank"], data["top_fraction"], strict=True
    ):
        axis_b.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"rank {int(rank)}; top {fraction:.1%}",
            va="center",
            fontsize=6,
        )
    axis_b.set_xlim(0, 1.08)
    axis_b.set_xlabel("HLA-G ligand availability")
    axis_b.set_title("HLA-G rank within each sender")
    final_edges = source[source["panel"].eq("B_final")]
    if not final_edges.empty:
        retained = int(final_edges["interaction_id"].nunique())
        maximum = float(final_edges["availability"].max())
        evt1 = final_edges[
            final_edges["sender"].eq("EVT_1") & final_edges["receiver"].eq("EVT_1")
        ]
        evt1_value = float(evt1["availability"].max()) if not evt1.empty else 0.0
        axis_b.text(
            0.0,
            -0.28,
            f"Final complete edge: {retained}/2 retained; "
            f"max={maximum:.4f}; EVT_1 self={evt1_value:.1f}",
            transform=axis_b.transAxes,
            fontsize=5.7,
        )
    _panel_label(axis_b, "B")

    data = source[source["panel"].eq("C")].copy()
    cluster_order = ["EVT_1", "EVT_2", "iEVT"]
    gene_order = TF_GENES
    values = data["mean_response"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    norm = mpl.colors.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
    cmap = mpl.colormaps["viridis"]
    for _, row in data.iterrows():
        x = gene_order.index(str(row["gene"]))
        y = cluster_order.index(str(row["receiver"]))
        value = (
            float(row["mean_response"]) if pd.notna(row["mean_response"]) else np.nan
        )
        if np.isfinite(value):
            axis_c.scatter(
                x,
                y,
                s=95,
                color=cmap(norm(value)),
                edgecolor=BLACK if bool(row["external_active_tf"]) else "none",
                linewidth=1.1,
            )
        else:
            axis_c.scatter(x, y, s=35, marker="x", color=MID_GRAY, linewidth=1.0)
        if bool(row["external_active_tf"]):
            axis_c.scatter(
                x, y, s=125, facecolors="none", edgecolors=BLACK, linewidth=1.0
            )
    axis_c.set_xticks(np.arange(len(gene_order)), gene_order, rotation=45, ha="right")
    axis_c.set_yticks(np.arange(len(cluster_order)), cluster_order)
    axis_c.invert_yaxis()
    axis_c.set_xlim(-0.6, len(gene_order) - 0.4)
    axis_c.set_title("Tutorial active-TF support")
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axis_c, fraction=0.05, pad=0.03)
    colorbar.set_label("Mean expression")
    axis_c.text(
        0.02,
        -0.28,
        "Black ring: external active TF; x: response not estimable",
        transform=axis_c.transAxes,
        fontsize=5.7,
    )
    _panel_label(axis_c, "C")

    data = source[source["panel"].eq("D")].copy()
    cell_types = ["FIB-A", "FIB-B", "FIB-P", "Basal", "Spinious", "DC", "Pericyte"]
    y = np.arange(len(cell_types))
    for context, color, offset in (("E13.5", BLUE, -0.12), ("E14.5", ORANGE, 0.12)):
        subset = (
            data[data["context"].eq(context)].set_index("cell_type").reindex(cell_types)
        )
        values = subset["cell_fraction"].to_numpy(dtype=float)
        axis_d.scatter(values, y + offset, color=color, s=35, label=context, zorder=3)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                axis_d.scatter(0, y[index] + offset, marker="x", color=MID_GRAY, s=30)
                axis_d.text(
                    0.006,
                    y[index] + offset,
                    "structural NA",
                    va="center",
                    fontsize=5.5,
                    color=MID_GRAY,
                )
    axis_d.set_yticks(y, cell_types)
    axis_d.invert_yaxis()
    axis_d.set_xlabel("Cell fraction")
    axis_d.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis_d.set_title("Embryonic composition (missing stage/type is not zero)")
    axis_d.legend(loc="lower right")
    _panel_label(axis_d, "D")

    data = source[source["panel"].eq("E")].copy()
    markers = {"E13.5": "o", "E14.5": "s"}
    colors = {"WNT": VERMILLION, "ncWNT": BLUE}
    for pathway in ("WNT", "ncWNT"):
        subset = (
            data[data["pathway"].eq(pathway)]
            .set_index("context")
            .reindex(["E13.5", "E14.5"])
        )
        axis_e.plot(
            [0, 1],
            subset["total_strength"],
            color=colors[pathway],
            linewidth=1.2,
            label=pathway,
        )
        for index, context in enumerate(("E13.5", "E14.5")):
            row = subset.loc[context]
            axis_e.scatter(
                index,
                row["total_strength"],
                s=20 + 1.7 * float(row["nonzero_edges"]),
                color=colors[pathway],
                marker=markers[context],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    axis_e.set_yscale("log")
    axis_e.set_xticks([0, 1], ["E13.5", "E14.5"])
    axis_e.set_ylabel("CellChat pathway total strength (log)")
    axis_e.set_title("WNT pathway reconfiguration")
    axis_e.legend(loc="best", title="Marker area = nonzero edges")
    _panel_label(axis_e, "E")

    figure.suptitle(
        "Trophoblast CRYCHIC support and embryonic CellChat reference-only",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.11)
    _save_figure(figure, figures_dir, "figure04_trophoblast_embryo")


def _synthetic_data(
    synthetic_root: Path,
    book: MetricBook,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = _read_json(synthetic_root / "summary.json")
    checks = pd.read_csv(synthetic_root / "checks.csv")
    scenarios = pd.read_csv(synthetic_root / "scenario_metrics.csv")
    edges = pd.read_csv(synthetic_root / "edge_metrics.csv")
    required_scenarios = {
        "active",
        "global_null",
        "abundance_only",
        "receiver_autonomous",
        "ligand_only",
        "target_only",
        "receptor_knockout",
    }
    if set(scenarios["scenario"]) != required_scenarios:
        raise ValueError(
            "synthetic scenario_metrics.csv does not contain the locked v0.1 scenarios"
        )
    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], ["scenario", "check"]]
        raise ValueError(
            f"synthetic benchmark contains failed checks: {failed.to_dict('records')}"
        )

    main_edges = edges[edges["interaction_id"].eq("CXCL10_CXCR3")].copy()
    main_edges = main_edges.merge(
        scenarios[
            [
                "scenario",
                "main_state_relative_change",
                "main_ecosystem_relative_change",
                "autonomous_response_effect_mean",
                "main_stim_receptor_availability",
                "main_stim_state",
            ]
        ],
        on="scenario",
        how="left",
        validate="one_to_one",
    )

    source_parts: list[pd.DataFrame] = []
    panel_a = (
        checks.groupby("scenario", sort=False)
        .agg(checks_total=("check", "size"), checks_passed=("passed", "sum"))
        .reset_index()
    )
    panel_a.insert(0, "panel", "A")
    panel_a["data_role"] = "contract_checks_by_scenario"
    source_parts.append(panel_a)

    panel_b = main_edges[
        [
            "scenario",
            "truth_active",
            "state_effect",
            "target_response_effect",
            "ecosystem_effect",
            "recovery_score",
        ]
    ].copy()
    panel_b.insert(0, "panel", "B")
    panel_b["data_role"] = "main_edge_state_response_separation"
    source_parts.append(panel_b)

    panel_c = main_edges[
        [
            "scenario",
            "truth_active",
            "recovery_score",
            "attribution_explained_fraction",
            "signed_residual_ratio",
        ]
    ].copy()
    panel_c.insert(0, "panel", "C")
    panel_c["data_role"] = "exploratory_recovery_score"
    source_parts.append(panel_c)

    diagnostic_rows = [
        {
            "diagnostic": "Abundance-only state relative change",
            "value": float(
                scenarios.loc[
                    scenarios["scenario"].eq("abundance_only"),
                    "main_state_relative_change",
                ].iloc[0]
            ),
            "scenario": "abundance_only",
            "unit": "relative_change",
        },
        {
            "diagnostic": "Abundance-only ecosystem relative change",
            "value": float(
                scenarios.loc[
                    scenarios["scenario"].eq("abundance_only"),
                    "main_ecosystem_relative_change",
                ].iloc[0]
            ),
            "scenario": "abundance_only",
            "unit": "relative_change",
        },
        {
            "diagnostic": "Receiver-autonomous state effect",
            "value": float(
                scenarios.loc[
                    scenarios["scenario"].eq("receiver_autonomous"),
                    "main_state_effect",
                ].iloc[0]
            ),
            "scenario": "receiver_autonomous",
            "unit": "availability_state_effect",
        },
        {
            "diagnostic": "KO stimulated receptor availability",
            "value": float(
                scenarios.loc[
                    scenarios["scenario"].eq("receptor_knockout"),
                    "main_stim_receptor_availability",
                ].iloc[0]
            ),
            "scenario": "receptor_knockout",
            "unit": "availability",
        },
        {
            "diagnostic": "KO stimulated interaction state",
            "value": float(
                scenarios.loc[
                    scenarios["scenario"].eq("receptor_knockout"),
                    "main_stim_state",
                ].iloc[0]
            ),
            "scenario": "receptor_knockout",
            "unit": "availability_state",
        },
    ]
    panel_d = pd.DataFrame(diagnostic_rows)
    panel_d.insert(0, "panel", "D")
    panel_d["data_role"] = "negative_control_diagnostics"
    source_parts.append(panel_d)
    source = pd.concat(source_parts, ignore_index=True, sort=False)

    edge_truth = summary["edge_truth_metrics"]
    book.add(
        section="synthetic_contract",
        dataset="Synthetic v0.1",
        metric="checks_passed",
        value=int(summary["checks_passed"]),
        unit=f"of_{int(summary['checks_total'])}",
        status="pass" if bool(summary["all_checks_passed"]) else "failed",
        method="CRYCHIC synthetic negative controls",
        resource="synthetic known truth",
        interpretation="Scenario-level contract checks; not inferential calibration.",
        source_figure="figure05_synthetic",
    )
    for metric_name, key in (
        ("edge_truth_auroc", "auroc"),
        ("edge_truth_average_precision", "average_precision"),
    ):
        book.add(
            section="synthetic_edge_truth",
            dataset="Synthetic v0.1",
            metric=metric_name,
            value=float(edge_truth[key]),
            unit="score",
            status="single_positive_smoke_only",
            reason_code="one_positive_edge_not_calibration",
            method="CRYCHIC synthetic negative controls",
            resource="synthetic known truth",
            interpretation=(
                f"n_positive={int(edge_truth['n_positive'])}; "
                f"n_negative={int(edge_truth['n_negative'])}; synthetic truth only."
            ),
            source_figure="figure05_synthetic",
        )
    for field in ("p_values", "q_values", "fdr"):
        book.add(
            section="synthetic_inference_guardrail",
            dataset="Synthetic v0.1",
            metric=field,
            value=None,
            value_text="null",
            unit="not_available",
            status="not_available",
            reason_code=str(summary["formal_inference"]["reason_code"]),
            method="CRYCHIC synthetic negative controls",
            interpretation="Formal inference is disabled in v0.1.",
            source_figure="figure05_synthetic",
        )
    for row in checks.itertuples(index=False):
        book.add(
            section="synthetic_check",
            dataset="Synthetic v0.1",
            metric=f"{row.scenario}:{row.check}",
            value=float(row.observed),
            unit="scenario_specific",
            status="pass" if bool(row.passed) else "failed",
            method="CRYCHIC synthetic negative controls",
            resource="synthetic known truth",
            interpretation=str(row.expectation),
            source_figure="figure05_synthetic",
        )

    scenario_index = scenarios.set_index("scenario")
    compact = {
        "summary": summary,
        "active": scenario_index.loc["active"].to_dict(),
        "global_null": scenario_index.loc["global_null"].to_dict(),
        "abundance_only": scenario_index.loc["abundance_only"].to_dict(),
        "receiver_autonomous": scenario_index.loc["receiver_autonomous"].to_dict(),
        "ligand_only": scenario_index.loc["ligand_only"].to_dict(),
        "target_only": scenario_index.loc["target_only"].to_dict(),
        "receptor_knockout": scenario_index.loc["receptor_knockout"].to_dict(),
    }
    return source, compact


def _plot_synthetic(source: pd.DataFrame, figures_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    scenario_order = [
        "active",
        "global_null",
        "abundance_only",
        "receiver_autonomous",
        "ligand_only",
        "target_only",
        "receptor_knockout",
    ]
    display = {
        "active": "Active",
        "global_null": "Global null",
        "abundance_only": "Abundance only",
        "receiver_autonomous": "Receiver autonomous",
        "ligand_only": "Ligand only",
        "target_only": "Target only",
        "receptor_knockout": "Receptor KO",
    }

    axis = axes[0, 0]
    data = source[source["panel"].eq("A")].set_index("scenario").reindex(scenario_order)
    y = np.arange(len(data))
    axis.barh(y, data["checks_total"], color=LIGHT_GRAY, label="Checks total")
    axis.barh(y, data["checks_passed"], color=GREEN, label="Passed")
    axis.set_yticks(y, [display[item] for item in scenario_order])
    axis.invert_yaxis()
    axis.set_xlabel("Contract checks")
    axis.set_title("All 14 scenario checks passed")
    axis.legend(loc="lower right")
    _panel_label(axis, "A")

    axis = axes[0, 1]
    data = (
        source[source["panel"].eq("B")]
        .set_index("scenario")
        .reindex(scenario_order)
        .reset_index()
    )
    for _, row in data.iterrows():
        color = VERMILLION if bool(row["truth_active"]) else BLUE
        axis.scatter(
            row["state_effect"],
            row["target_response_effect"],
            color=color,
            s=36,
            zorder=3,
        )
        axis.annotate(
            display[str(row["scenario"])],
            (row["state_effect"], row["target_response_effect"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5.8,
        )
    axis.axhline(0, color=LIGHT_GRAY, linewidth=0.8)
    axis.axvline(0, color=LIGHT_GRAY, linewidth=0.8)
    axis.set_xlabel("CXCL10-CXCR3 availability-state effect")
    axis.set_ylabel("Target response effect (log1p CPM)")
    axis.set_title("State and receiver response remain separable")
    _panel_label(axis, "B")

    axis = axes[1, 0]
    data = (
        source[source["panel"].eq("C")]
        .set_index("scenario")
        .reindex(scenario_order)
        .reset_index()
    )
    colors = [VERMILLION if bool(value) else BLUE for value in data["truth_active"]]
    bars = axis.barh(
        np.arange(len(data)),
        data["recovery_score"],
        color=colors,
    )
    axis.set_yticks(np.arange(len(data)), [display[item] for item in scenario_order])
    axis.invert_yaxis()
    axis.set_xlim(0, 0.62)
    axis.set_xlabel("Exploratory recovery score")
    axis.set_title("Positive control exceeds negative controls")
    for bar, value in zip(bars, data["recovery_score"], strict=True):
        axis.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=6,
        )
    _panel_label(axis, "C")

    axis = axes[1, 1]
    data = source[source["panel"].eq("D")].copy()
    data["short_label"] = [
        "Abundance\nstate rel.",
        "Abundance\necosystem rel.",
        "Autonomous\nstate effect",
        "KO stim\nreceptor",
        "KO stim\nstate",
    ]
    x = np.arange(len(data))
    colors = [BLUE if value >= 0 else ORANGE for value in data["value"]]
    axis.bar(x, data["value"], color=colors)
    axis.axhline(0, color=BLACK, linewidth=0.8)
    axis.set_xticks(x, data["short_label"])
    axis.set_ylabel("Dimensionless diagnostic value")
    axis.set_title("Abundance sensitivity and knockout gates")
    for index, value in enumerate(data["value"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.006 if value >= 0 else -0.006
        axis.text(index, value + offset, f"{value:.3f}", ha="center", va=va, fontsize=6)
    _panel_label(axis, "D")

    figure.suptitle(
        "Synthetic v0.1 negative controls (known truth; not calibration)",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.90,
        bottom=0.11,
        hspace=0.42,
        wspace=0.32,
    )
    _save_figure(figure, figures_dir, "figure05_synthetic")


def _resource_records(repo_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((repo_root / "resources").glob("*.json")):
        manifest = _read_json(path)
        records.append(
            {
                "manifest": path.relative_to(repo_root).as_posix(),
                "resource_id": manifest.get("resource_id", ""),
                "version": manifest.get("version", ""),
                "species": manifest.get("species", ""),
                "gene_namespace": manifest.get("gene_namespace", ""),
                "license": manifest.get("license", ""),
                "adapter_version": manifest.get("adapter_version", ""),
                "retrieved_at": manifest.get("retrieved_at", ""),
                "manifest_sha256": _sha256(path),
            }
        )
    return records


def _canonical_result_records(
    canonical_root: Path,
    workspace_root: Path,
    book: MetricBook,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = canonical_root / "summary.json"
    expected = {
        "kang2018": "Kang IFN-beta",
        "ad_skin": "AD skin",
        "trophoblast": "Trophoblast",
    }
    if summary_path.is_file():
        summary = _read_json(summary_path)
        declared = {
            str(item["name"]): item
            for item in summary.get("datasets", [])
            if isinstance(item, dict) and "name" in item
        }
    else:
        summary = {
            "schema_version": "canonical-v0.1-summary",
            "status": "missing",
            "wall_seconds": None,
            "datasets": [],
        }
        declared = {}

    records: list[dict[str, Any]] = []
    for name, display_name in expected.items():
        declared_record = declared.get(name, {})
        relative_metrics = str(declared_record.get("metrics", f"{name}/metrics.json"))
        metrics_path = canonical_root / relative_metrics
        if metrics_path.is_file():
            metrics = _read_json(metrics_path)
            status = str(metrics.get("status", "unknown"))
            timing = metrics.get("timing_seconds", {})
            result = metrics.get("result") or {}
            table_rows = {
                str(key): int(value)
                for key, value in (result.get("table_rows") or {}).items()
            }
            failure = metrics.get("failure") or {}
            inference_reason = str(
                (metrics.get("formal_inference") or {}).get(
                    "reason_code", "v0_1_inferential_disabled"
                )
            )
            if status == "complete":
                reason_code = inference_reason
            elif status == "dry_run_complete":
                reason_code = "dry_run_only_no_result_tables"
            elif status == "failed":
                reason_code = str(failure.get("type", "runner_failed"))
            else:
                reason_code = "runner_status_unknown"
            resource = metrics.get("resources", {}).get("ligand_receptor") or {}
            resource_label = (
                f"{resource.get('resource_id', 'unknown')} "
                f"{resource.get('version', 'unknown')}"
            )
            target_prior = metrics.get("resources", {}).get("target_prior") or {}
            selection = metrics.get("interaction_selection") or {}
            lineage = metrics.get("analysis_lineage") or {}
            threadpool_limit = (metrics.get("threading") or {}).get("threadpool_limit")
            execution = metrics.get("execution") or {}
            memory = metrics.get("memory_measurement") or {}
            isolated = metrics.get("isolated_process", memory.get("isolated_process"))
            peak_rss_scope = (
                memory.get("peak_rss_scope")
                or metrics.get("peak_rss_scope")
                or execution.get("peak_rss_scope")
                or execution.get("process_scope")
                or (
                    "standalone_isolated_process"
                    if isolated is True
                    else "sequential_process_upper_bound"
                )
            )
            record = {
                "run_id": f"canonical_v01_{name}",
                "dataset": display_name,
                "status": status,
                "reason_code": reason_code,
                "inferential_fields_enabled": bool(
                    (metrics.get("formal_inference") or {}).get("enabled", False)
                ),
                "resource": resource_label,
                "seed": (metrics.get("config") or {}).get(
                    "random_seed", "not_recorded"
                ),
                "threads": (
                    f"BLAS<={int(threadpool_limit)}"
                    if threadpool_limit is not None
                    else "not_recorded"
                ),
                "hardware": (metrics.get("environment") or {}).get(
                    "platform", "not_recorded"
                ),
                "result_directory": _relative(canonical_root / name, workspace_root),
                "wall_seconds": _native(timing.get("wall")),
                "fit_and_persist_seconds": _native(timing.get("fit_and_persist")),
                "peak_rss_mb": _native(metrics.get("peak_rss_mb")),
                "peak_rss_scope": str(peak_rss_scope),
                "mode": result.get("mode", "not_available"),
                "max_interactions": _native(selection.get("max_interactions")),
                "resource_truncated": bool(selection.get("resource_truncated", False)),
                "selection_rule": str(selection.get("selection_rule", "not_available")),
                "target_prior": (
                    f"{target_prior.get('resource_id')} {target_prior.get('version')}"
                    if target_prior
                    else "not_provided"
                ),
                "analysis_lineage_digest": str(lineage.get("digest", "not_available")),
                "table_rows": table_rows,
                "result_stages": list(result.get("stages", [])),
                "result_warnings": list(result.get("warnings", [])),
                "failure": failure or None,
            }
        else:
            record = {
                "run_id": f"canonical_v01_{name}",
                "dataset": display_name,
                "status": "missing",
                "reason_code": "metrics_json_missing",
                "inferential_fields_enabled": False,
                "resource": "not_available",
                "seed": "not_recorded",
                "threads": "not_recorded",
                "hardware": "not_recorded",
                "result_directory": _relative(canonical_root / name, workspace_root),
                "wall_seconds": None,
                "fit_and_persist_seconds": None,
                "peak_rss_mb": None,
                "peak_rss_scope": "not_available",
                "mode": "not_available",
                "max_interactions": None,
                "resource_truncated": False,
                "selection_rule": "not_available",
                "target_prior": "not_available",
                "analysis_lineage_digest": "not_available",
                "table_rows": {},
                "result_stages": [],
                "result_warnings": [],
                "failure": None,
            }
        records.append(record)
        book.add(
            section="canonical_interaction_selection",
            dataset=display_name,
            metric="selected_interaction_limit",
            value=record["max_interactions"],
            value_text=str(record["selection_rule"]),
            unit="interactions",
            status=(
                "resource_truncated_v0_1_smoke"
                if record["resource_truncated"]
                else "full_resource_or_not_available"
            ),
            reason_code=(
                "deterministic_outcome_independent_prefilter"
                if record["resource_truncated"]
                else str(record["reason_code"])
            ),
            method="CRYCHIC canonical v0.1 runner",
            resource=str(record["resource"]),
            interpretation="A truncated deterministic interaction selection does not establish full-resource coverage.",
        )
        book.add(
            section="canonical_result_run",
            dataset=display_name,
            metric="runner_status",
            value_text=str(record["status"]),
            unit="status",
            status=str(record["status"]),
            reason_code=str(record["reason_code"]),
            method="CRYCHIC canonical v0.1 runner",
            resource=str(record["resource"]),
            interpretation=(
                "Schema/provenance execution status; biology panels use locked "
                "sources, with AD quantitative panels switching to corrected final "
                "interactions when complete."
            ),
        )
        for metric_name, value, unit in (
            ("wall_seconds", record["wall_seconds"], "seconds"),
            (
                "fit_and_persist_seconds",
                record["fit_and_persist_seconds"],
                "seconds",
            ),
            ("peak_rss_mb", record["peak_rss_mb"], "MiB"),
        ):
            book.add(
                section="canonical_result_run",
                dataset=display_name,
                metric=metric_name,
                value=value,
                unit=unit,
                status=("observed_runtime" if value is not None else "not_available"),
                reason_code=("" if value is not None else str(record["reason_code"])),
                method="CRYCHIC canonical v0.1 runner",
                resource=str(record["resource"]),
                interpretation=(
                    f"Runner-level diagnostic; peak_rss_scope="
                    f"{record['peak_rss_scope']}; not a cross-machine comparison."
                ),
            )
        for table_name, row_count in record["table_rows"].items():
            book.add(
                section="canonical_result_tables",
                dataset=display_name,
                metric=f"{table_name}_rows",
                value=row_count,
                unit="rows",
                status="persisted",
                method="CRYCHIC canonical v0.1 runner",
                resource=str(record["resource"]),
                interpretation="Rows declared in the persisted result manifest.",
            )
    payload = {
        "summary": summary,
        "datasets": records,
    }
    return payload, records


def _run_records(
    metrics: dict[str, dict[str, Any]],
    liana_manifest: dict[str, Any],
    synthetic_summary: dict[str, Any],
    synthetic_result_directory: str,
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": "crychic_kang_prelim",
            "dataset": "Kang IFN-beta",
            "status": "success_preliminary",
            "reason_code": "availability_and_response_only",
            "inferential_fields_enabled": bool(
                metrics["kang"]["inferential_fields_enabled"]
            ),
            "resource": "CellChatDB human v2",
            "seed": "not_recorded",
            "threads": "not_recorded",
            "hardware": "not_recorded",
            "result_directory": "benchmark_work/crychic_kang_prelim",
        },
        {
            "run_id": "crychic_ad_prelim",
            "dataset": "AD skin",
            "status": "exploratory_only",
            "reason_code": "normalized_only_and_unpaired_conditions",
            "inferential_fields_enabled": bool(
                metrics["ad"]["inferential_fields_enabled"]
            ),
            "resource": "CellChatDB human v2",
            "seed": "not_recorded",
            "threads": "not_recorded",
            "hardware": "not_recorded",
            "result_directory": "benchmark_work/crychic_ad_prelim",
        },
        {
            "run_id": "crychic_trophoblast_prelim",
            "dataset": "Trophoblast",
            "status": "single_context_no_differential_contrast",
            "reason_code": str(metrics["trophoblast"]["reason_code"]),
            "inferential_fields_enabled": False,
            "resource": "CellPhoneDB human v5.0.0",
            "seed": "not_recorded",
            "threads": "not_recorded",
            "hardware": "not_recorded",
            "result_directory": "benchmark_work/crychic_trophoblast_prelim",
        },
        {
            "run_id": "crychic_embryonic_skin",
            "dataset": "Embryonic skin",
            "status": "not_run_reference_only",
            "reason_code": "merged_stage_objects_without_subject_replicates",
            "inferential_fields_enabled": False,
            "resource": "CellChat archived reference",
            "seed": "not_applicable",
            "threads": "not_applicable",
            "hardware": "not_recorded",
            "result_directory": "benchmark_work/cellchat_reference/embryo_E13; benchmark_work/cellchat_reference/embryo_E14",
        },
        {
            "run_id": "liana_kang_default",
            "dataset": "Kang IFN-beta",
            "status": "success_rank_only",
            "reason_code": "default_resource_not_harmonized",
            "inferential_fields_enabled": False,
            "resource": str(liana_manifest["resource"]),
            "seed": liana_manifest["seed"],
            "threads": liana_manifest["n_jobs"],
            "hardware": "not_recorded",
            "result_directory": "benchmark_work/liana_kang",
        },
        {
            "run_id": "liana_kang_harmonized_resource",
            "dataset": "Kang IFN-beta",
            "status": "not_run",
            "reason_code": "harmonized_resource_output_not_available",
            "inferential_fields_enabled": False,
            "resource": "pending",
            "seed": "not_applicable",
            "threads": "not_applicable",
            "hardware": "not_applicable",
            "result_directory": "not_available",
        },
        {
            "run_id": "synthetic_v01",
            "dataset": "Synthetic v0.1",
            "status": "complete",
            "reason_code": "single_positive_smoke_not_calibration",
            "inferential_fields_enabled": False,
            "resource": "synthetic known truth",
            "seed": synthetic_summary["seed"],
            "threads": "not_recorded",
            "hardware": "not_recorded",
            "result_directory": synthetic_result_directory,
        },
    ]


def _format_metric(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def _report_markdown(
    *,
    generated_at: str,
    environment: dict[str, Any],
    dataset_specs: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    kang: dict[str, Any],
    ad_liana: dict[str, Any],
    troph_embryo: dict[str, Any],
    synthetic: dict[str, Any],
    canonical: dict[str, Any],
) -> str:
    comp = kang["composition"]
    ad_inflam = ad_liana["inflammatory"]
    nl_mif = ad_liana["pathway_stats"]["NL"]
    ls_mif = ad_liana["pathway_stats"]["LS"]
    ad_corrected = (
        ad_liana["crychic_output_source"] == "corrected_canonical_v01_interactions"
    )
    if ad_corrected:
        ad_executive = (
            "**AD 炎症组成和 MIF 参考被支持性复现。** "
            "组成来自 archived CellChat reference-only 对象；MIF 定量比较来自"
            "修复 normalized-only detection 语义后重新生成的 CRYCHIC canonical "
            f"results。MIF 在 CRYCHIC availability 中 NL/LS 分别排第 "
            f"{int(nl_mif['crychic_mif_rank'])} 和第 "
            f"{int(ls_mif['crychic_mif_rank'])}，sender-receiver 排名相关分别为 "
            f"**{nl_mif['mif_edge_spearman']:.3f}** 和 "
            f"**{ls_mif['mif_edge_spearman']:.3f}**。"
        )
        ad_source_statement = (
            "Figure 3B/C 与下述 CRYCHIC MIF 数值来自 corrected canonical v0.1 "
            "`interactions.parquet`；archived CellChat 仅作为 reference comparator。"
        )
    else:
        ad_executive = (
            "**AD archived CellChat reference 的炎症组成方向被恢复，但 CRYCHIC "
            "MIF 定量结论暂不生效。** 当前仅有受 normalized-only detection shrink "
            "bug 影响的 preliminary availability，相关指标标为 "
            "`invalidated_pending_corrected_rerun`。"
        )
        ad_source_statement = (
            "Figure 3B/C 当前读取 preliminary normalized-only availability，已知会受 "
            "detection shrink bug 影响，因此仅保留为待替换诊断，不构成复现结论。"
        )
    hla_response = {item["receiver"]: item for item in troph_embryo["hla_response"]}
    hla_ligand = {item["sender"]: item for item in troph_embryo["hla_ligand"]}
    wnt = {(item["context"], item["pathway"]): item for item in troph_embryo["wnt"]}
    synthetic_summary = synthetic["summary"]
    synthetic_truth = synthetic_summary["edge_truth_metrics"]
    canonical_records = canonical["datasets"]
    canonical_by_dataset = {item["dataset"]: item for item in canonical_records}
    kang_final = kang["canonical_final"]
    if kang_final["status"] == "complete":
        kang_final_statement = (
            "Final canonical 保留的同一 CXCL10-CXCR3 row 在 ctrl/stim 的 availability "
            f"分别为 {kang_final['ctrl_availability']:.6f} / "
            f"{kang_final['stim_availability']:.6f}（delta="
            f"{kang_final['availability_delta']:.6f}）。这是经过完整 exploratory branches 后"
            "仍进入结果表的 retention diagnostic，不替代 8/8 donor 的纯 availability "
            f"estimand {kang['cxcl10_effect']:.6f}。最终 response audit 在所有 8 个观察到的 "
            f"receiver 中有 {kang_final['estimable_receivers']}/"
            f"{kang_final['observed_receivers']} 可估，"
            f"{kang_final['positive_ok_isg_rows']} 个可估 ISG rows 全为正；"
            f"{kang_final['not_estimable_isg_rows']} 个 rows 为 NE。"
        )
        kang_strength_statement = (
            f"Final interactions 中 {kang_final['finite_strength_rows']:,} 行有有限 "
            f"integrated strength，其中 {kang_final['positive_strength_rows']:,} 行大于 0；"
            f"该 CXCL10 edge 的 integrated strength 为 "
            f"{kang_final['cxcl10_integrated_strength']:.1f}，且 comm_probability 非空行为 "
            f"{kang_final['comm_probability_nonnull_rows']}。因此 availability 支持不能被写成"
            "该 edge 已获 integrated attribution 或校准概率支持。"
        )
    else:
        kang_final_statement = (
            "Final canonical Kang diagnostic 尚不可用；Figure 2 仅报告锁定的 preliminary "
            "pure-availability estimand。"
        )
        kang_strength_statement = (
            "Final integrated-strength/probability audit 尚不可用，不作替代性推断。"
        )
    hla_final = troph_embryo["hla_final_complete_edge"]
    if hla_final["available"]:
        hla_final_statement = (
            f"Final CellPhoneDB top-selection 只保留 "
            f"{hla_final['retained_interactions']}/{hla_final['resource_interactions']} 个 HLA-G "
            "resource interactions；最大 complete-edge state availability 为 "
            f"{hla_final['max_complete_edge_availability']:.6f}（"
            f"`{hla_final['max_complete_edge_label']}`），EVT_1 -> EVT_1 为 "
            f"{hla_final['evt1_self_availability']:.6f}。因此 HLA-G ligand/expression 支持"
            "没有转化为强 EVT complete edge。"
        )
    else:
        hla_final_statement = (
            "Final trophoblast complete-edge audit 尚不可用；HLA-G 只作 ligand/expression "
            "支持性描述。"
        )

    performance_parts = []
    for dataset_name, workload in (
        ("Kang IFN-beta", "LR selection + target-prior attribution/scoring"),
        ("AD skin", "availability expansion"),
    ):
        record = canonical_by_dataset.get(dataset_name, {})
        if record.get("status") != "complete":
            continue
        sample_score_rows = int(record.get("table_rows", {}).get("sample_scores", 0))
        prior_text = (
            f"，target prior=`{record['target_prior']}`"
            if record.get("target_prior") not in {None, "not_provided", "not_available"}
            else ""
        )
        performance_parts.append(
            f"{dataset_name} 的 top-{int(record['max_interactions'])} {workload}"
            f"{prior_text} 用时 {_format_metric(record['wall_seconds'])} s，并持久化 "
            f"{sample_score_rows:,} 行 sample scores"
        )
    performance_statement = (
        "；".join(performance_parts)
        if performance_parts
        else "Final Kang/AD performance records 尚不完整"
    )
    kang_canonical = canonical_by_dataset.get("Kang IFN-beta", {})
    if kang_canonical.get("resource_truncated"):
        kang_selection_statement = (
            "Kang final interaction selection status=`resource_truncated_v0_1_smoke`；"
            f"top-{int(kang_canonical['max_interactions'])} 使用 deterministic、"
            "context-label/outcome-independent rule："
            f"`{kang_canonical['selection_rule']}`。该运行只验证截断后的 v0.1 "
            "executable slice，不代表 full-resource coverage。"
        )
    else:
        kang_selection_statement = (
            "Kang final interaction-selection provenance 不完整，不能声明 full-resource "
            "coverage。"
        )

    dataset_lines = []
    for item in dataset_specs:
        source_subjects = "NA" if item["subjects"] is None else str(item["subjects"])
        final_scale = (
            "NA"
            if item["final_analysis_cells"] is None
            else f"{int(item['final_analysis_cells']):,} / {int(item['final_cell_types'])}"
        )
        final_units = (
            "NA"
            if item["final_analysis_units"] is None
            else (
                f"{int(item['final_analysis_units'])} / {int(item['final_subjects'])}"
            )
        )
        dataset_lines.append(
            f"| {item['dataset']} | {int(item['cells']):,} / "
            f"{item['cell_types']} | {final_scale} | "
            f"{item['analysis_units']} / {source_subjects} | {final_units} | "
            f"{item['contexts']} | {item['input_mode']} | "
            f"{item['design_status']} |"
        )
    resource_lines = [
        f"| {item['resource_id']} | {item['version']} | {item['species']} | {item['gene_namespace']} | {item['adapter_version']} | {item['license']} | `{item['manifest_sha256'][:16]}` |"
        for item in resources
    ]
    run_lines = [
        f"| `{item['run_id']}` | {item['dataset']} | `{item['status']}` | `{item['reason_code']}` | `{item['result_directory']}` | {item['resource']} | {item['seed']} | {item['threads']} |"
        for item in runs
    ]
    input_lines = [
        f"| `{item['path']}` | {item['role']} | {int(item['bytes']):,} | `{item['sha256'][:16]}` |"
        for item in inputs
    ]
    canonical_lines = []
    for item in canonical_records:
        table_rows = item["table_rows"]
        table_text = (
            ", ".join(f"{name}={count:,}" for name, count in sorted(table_rows.items()))
            if table_rows
            else "none"
        )
        canonical_lines.append(
            f"| {item['dataset']} | `{item['status']}` | "
            f"{_format_metric(item['wall_seconds'])} | "
            f"{_format_metric(item['fit_and_persist_seconds'])} | "
            f"{_format_metric(item['peak_rss_mb'], 1)} | "
            f"`{item['peak_rss_scope']}` | "
            f"`{item['mode']}` | {table_text} | `{item['reason_code']}` |"
        )

    text = f"""# CRYCHIC 经典单细胞互作数据集 v0.1 探索性 Benchmark 汇总

报告 ID：`{REPORT_ID}`

生成时间：`{generated_at}`

分析范围：真实数据的输入校验、pseudobulk response、配体-受体 availability、外部参考复现与默认资源内排名稳定性，以及 v0.1 synthetic known-truth negative controls。生物学 panels 以 response/availability 为主；Kang final canonical 的 integrated strength 与 sender coupling 仅作为 v0.1 exploratory in-sample diagnostics，calibrated resampling 与 formal inference 仍不在本版声明范围内。

## 执行摘要

1. **Kang 2018 配对 IFN-beta 数据跑通。** 8 名 donor 的 ctrl/stim 均被保留。细胞组成 ctrl/stim 的 donor 内 Spearman 中位数为 **{comp["median_subject_spearman"]:.3f}**，全体 donor-cell-type 的绝对比例变化中位数为 **{comp["median_absolute_fraction_change"]:.4f}**，达到预锁定阈值（rho >= 0.85；中位变化 <= 0.05）。
2. **已知 IFN 受体程序被恢复。** 在预注册的 7 个 major receivers 中有 6/7 可估，12 个 ISG 共 72 个可估 receiver-gene 组合全部为正向，正向比例 **{kang["positive_fraction"]:.1%}**；donor 方向一致性中位数 **{kang["median_consistency"]:.1%}**。Dendritic cells 因部分配对支持而不可估，未被填零。CXCL10-CXCR3 最强 8/8 donor 纯 availability 边为 `{kang["cxcl10_label"]}`，平均 stim-ctrl effect 为 **{kang["cxcl10_effect"]:.3f}**。{kang_final_statement}
3. **外源 IFN-beta 解释护栏成立。** availability 表中 endogenous IFNB1 支持行数为 **{kang["ifnb_rows"]}**；这与重组 IFN-beta 外加实验相容，不能把强 ISG response 反推为 endogenous sender 因果证据。
4. {ad_executive}
5. **LIANA 结果只支持排名稳定性描述。** 默认 consensus resource 下，同一 donor 的 ctrl/stim shared-edge magnitude rank Spearman 中位数为 **{ad_liana["liana_median_rho"]:.3f}**，top-500 Jaccard 中位数为 **{ad_liana["liana_median_jaccard"]:.3f}**。该资源没有与 CRYCHIC harmonize，因此不能据此声称跨方法优劣。
6. **Trophoblast HLA-G 与教程 TF 标签获得支持。** EVT_1 和 iEVT 的 HLA-G mean response 分别为 **{hla_response["EVT_1"]["mean_response"]:.3f}** 和 **{hla_response["iEVT"]["mean_response"]:.3f}**；HLA-G ligand availability 在各 sender 内分别处于 top **{hla_ligand["EVT_1"]["top_fraction"]:.1%}** 和 **{hla_ligand["iEVT"]["top_fraction"]:.1%}**。5 个预注册 TF 在 EVT_1/EVT_2/iEVT 的 15 个 cluster-TF 组合中有 {troph_embryo["tf_external_pairs"]} 个被 external active-TF 文件标注；该标签是 silver-standard 支持，不是独立真值。
7. **Embryonic skin 仅作无重复参考。** E13.5 的 DC 与 Pericyte 记为 structural absence，而不是表达量 0。WNT total strength 的 E14.5/E13.5 比值为 **{wnt[("E14.5", "WNT")]["total_strength"] / wnt[("E13.5", "WNT")]["total_strength"]:.3f}**；ncWNT 总强度下降，但非零边由 {int(wnt[("E13.5", "ncWNT")]["nonzero_edges"])} 增至 {int(wnt[("E14.5", "ncWNT")]["nonzero_edges"])}，支持“重连”而非简单整体上调。
8. **Synthetic negative controls 14/14 通过。** synthetic-only edge AUROC/AP 均为 **1.000**，但只有 {int(synthetic_truth["n_positive"])} 个 positive edge 和 {int(synthetic_truth["n_negative"])} 个 negative edges，因此它只是 smoke discrimination，不是 calibration。`p`、`q`、FDR 均为 null。

## 统计解释边界

- 这些真实数据**没有 edge-level ground truth**，因此不报告 edge AUROC、AUPRC、precision/recall 或“真阳性发现率”。CellChat archived score 与教程标签均是 supportive silver standard，不是独立真值概率。
- AUROC、AUPRC/AP、FDR 和 edge-recovery coverage 只允许来自带已知 truth 的 synthetic benchmark。Kang 的 6/7 `isg_receiver_coverage` 只是 receiver 可估性覆盖，不是 edge truth-recovery coverage。
- 本版不输出、也不从 rank/z-score 伪造 `p`、`q` 或 calibrated probability。图中没有显著性星号。response 的 standard error/z-score 仅是诊断字段，本报告不将其解释为正式推断。
- AD 输入只有上游 normalized `@data`，且 NL/LS 是不同 subject，故仅作 unpaired exploratory 描述；不能声称配对差异或校准显著性。
- Embryonic skin 每个 stage 只有一个 merged object，没有 subject replicate；只能展示组成和 CellChat reference 的 stage reconfiguration。
- Trophoblast 只有一个可验证 context，source sample 不能被冒充为 repeated-context subject，因此 differential contrast 明确为不可估。
- LIANA 当前只有默认 consensus resource 结果；缺少 harmonized LR resource 对照，资源效应与算法效应不能拆分。

## DEVELOPMENT_PLAN 合规状态

| Gate / slice | 当前状态 | 已覆盖内容与剩余缺口 |
|---|---|---|
| G0 | `mostly_implemented_not_signed_off` | 工程骨架、typed contracts、资源 manifest/checksum、基础测试已实现；但 CI quality gate 与独立统计签核尚未建立，因此不能声明 G0 正式通过。 |
| v0.1 executable slice | `implemented_experimental` | 覆盖 validation、pseudobulk、context graph、global/local contrasts、gene response、CellChat/CellPhoneDB/NicheNet resources、availability，以及 experimental attribution、sender assignment、scoring、versioned results 与 API。 |
| G1 | `NOT_PASSED` | Factorial EMM、完整 design-rank auditing 与独立统计 review 仍不完整；canonical run 受资源截断，且尚无 v0.3 calibration。 |
| Release | `not_released` | 本报告是开发态 benchmark，不代表 v0.1 release，不代表 G1 complete，也不提供校准显著性保证。 |

因此，本报告可证明一个受约束的 v0.1 executable slice 能在经典数据和 synthetic controls 上运行并保留失败语义；它不能替代 DEVELOPMENT_PLAN 中后续 gate 的统计验证与 release sign-off。

## 数据集与运行规模

| 数据集 | Source/prelim cells / types | Final analyzed cells / types | Source samples / subjects | Final samples / subjects | Contexts | Input | Design status |
|---|---:|---:|---:|---:|---:|---|---|
{chr(10).join(dataset_lines)}

Trophoblast 的 source/preliminary supportive biology 使用 3,312 cells / 42 cell types；final canonical runner 按配置只分析声明的 12 cell types（815 cells），并保存 subset lineage digest `{canonical_by_dataset.get("Trophoblast", dict()).get("analysis_lineage_digest", "not_available")}`。两者不得混称为同一分析规模。

![Figure 1](figures/figure01_overview.png)

**Figure 1.** 数据规模、subject 支持、final dry-run gene mapping、经 pooled support/top-K 后进入 final result 的 unique interactions，以及 preliminary compute 与 final canonical wall time。Panel C 不再使用受 normalized-only bug 影响的 AD preliminary pooled-support count。Final marker 只在 dataset status=`complete` 时绘制；runtime 不能作为跨机器性能结论，peak RSS scope 见 canonical runner 表。源数据：[figure01_overview.csv](source_data/figure01_overview.csv)。

## Kang 2018：配对 IFN-beta

预注册生物学来自 `kang2018_ifnb_supportive_v1`。组成稳定阈值和 ISG 方向阈值均在 response scoring 前锁定。预注册 major receiver set 中 6/7 个完整可估 receiver 为：{", ".join(kang["estimable_receivers"])}。Final all-observed audit 还包含 Megakaryocytes，因此其分母是 8，完整可估覆盖为 6/8；Dendritic cells 与 Megakaryocytes 均保留为 `not_estimable`，未填零。

抗原呈递基因 B2M/HLA-A/HLA-B/HLA-C/TAP1/TAP2 的可估 receiver-gene 正向比例为 **{kang["antigen_positive"]:.1%}**。availability 的 top edges 主要包括 CCL、GALECTIN 和 CXCL10-CXCR3；它们是 molecular availability 的配对效应，不是 active probability，也不等同于已证实的细胞间因果作用。

{kang_final_statement} {kang_strength_statement}

![Figure 2](figures/figure02_kang.png)

**Figure 2.** A，12 个 ISG 的 paired stim-ctrl log1p(CPM) response；NE 表示设计不可估。B，逐 donor 组成相关，虚线为预锁定最低值。C，要求 8/8 donor 完整配对后的 top pure availability-state edges，条末数字是方向一致性。D，preliminary pure-availability CXCL10-CXCR3 donor effects；文字另列 final-retained diagnostic，二者不是同一 estimand。源数据：[figure02_kang.csv](source_data/figure02_kang.csv)。

## AD skin 与 LIANA 排名比较

AD 三类炎症细胞比例如下：Inflam. FIB {ad_inflam["Inflam. FIB"]["NL"]:.3%} -> {ad_inflam["Inflam. FIB"]["LS"]:.3%}，Inflam. DC {ad_inflam["Inflam. DC"]["NL"]:.3%} -> {ad_inflam["Inflam. DC"]["LS"]:.3%}，Inflam. TC {ad_inflam["Inflam. TC"]["NL"]:.3%} -> {ad_inflam["Inflam. TC"]["LS"]:.3%}。这些是 condition-level composition 描述，不是 paired subject effect。

{ad_source_statement}

在所有与 archived CellChat 共有的 pathway 中，CRYCHIC availability 与 CellChat pathway order 的 Spearman 为 NL **{nl_mif["rank_spearman"]:.3f}**、LS **{ls_mif["rank_spearman"]:.3f}**。MIF edge concordance 使用完全匹配的 context/pathway/sender/receiver 粒度，shared edges 分别为 {int(nl_mif["mif_shared_edges"])} 和 {int(ls_mif["mif_shared_edges"])}。相关性衡量排序复现，不能把 CellChat strength 当成 ground-truth probability。

![Figure 3](figures/figure03_ad_liana.png)

**Figure 3.** A，archived CellChat AD composition（reference-only）。B，CellChat reference 与 CRYCHIC MIF sender-receiver 排名比较。C，共有 pathway 排名。D，LIANA default-resource 在同一 donor 两个 context 间的 rank stability；未与 CRYCHIC 做资源未统一的“准确率”比较。源数据：[figure03_ad_liana.csv](source_data/figure03_ad_liana.csv)。

## Trophoblast 与 embryonic skin

HLA-G 在 EVT_1、iEVT、eEVT 中分别位于各 receiver 全基因 response 的 top {hla_response["EVT_1"]["top_fraction"]:.2%}、{hla_response["iEVT"]["top_fraction"]:.2%}、{hla_response["eEVT"]["top_fraction"]:.2%}。CellPhoneDB 将 HLA-G 显示实体记为 UniProt `P17693`；生成器通过 checksum-pinned genes/interactions 表映射回 HLA-G，并定位 `CPI-SS05CE87F88` / `CPI-SS0DBA81CCF`。HLA-G ligand availability 高不保证任意 receiver 都有完整 receptor 支持，因此报告把 ligand rank 与 complete edge state 分开。

{hla_final_statement}

EVT_2 没有达到 response eligibility 的样本，故 expression/TF response 显示为 NE；external active-TF 标签仍原样保留。外部 DEG/TF 文件中的 P.Value/adj.P.Val 没有进入本报告指标或图源数据。

Embryonic skin 的 WNT top-20 sender-receiver Jaccard 为 **{wnt[("E13.5", "WNT")]["top20_sender_receiver_jaccard"]:.3f}**，ncWNT 为 **{wnt[("E13.5", "ncWNT")]["top20_sender_receiver_jaccard"]:.3f}**。这表明经典 WNT edge order 改变更明显；由于没有 subject replicate，只能称为 reference reconfiguration。

![Figure 4](figures/figure04_trophoblast_embryo.png)

**Figure 4.** A，HLA-G receiver expression。B，HLA-G ligand availability sender 内排名。C，教程 active-TF 支持；黑圈是 external label，x 是 response 不可估。D，embryonic stage composition；structural NA 不填零。E，WNT/ncWNT total strength 与非零边数量。源数据：[figure04_trophoblast_embryo.csv](source_data/figure04_trophoblast_embryo.csv)。

## Synthetic known-truth negative controls

Synthetic v0.1 使用 8 个 subjects、每 sample 平均 {int(synthetic_summary["mean_cells_per_sample"])} 个 cells，seed={int(synthetic_summary["seed"])}。七个 locked scenarios 为 active、global null、abundance only、receiver autonomous、ligand only、target only 和 receptor knockout；14 个预定义 contract checks 全部通过。

- Active positive control：CXCL10-CXCR3 state effect **{synthetic["active"]["main_state_effect"]:.4f}**、target response **{synthetic["active"]["truth_response_effect_mean"]:.4f}**、exploratory recovery score **{synthetic["active"]["main_recovery_score"]:.4f}**。
- Global null：recovery score **{synthetic["global_null"]["main_recovery_score"]:.4f}**，没有把 null edge 提升为 positive-control recovery。
- Abundance only：state relative change **{synthetic["abundance_only"]["main_state_relative_change"]:+.2%}**，composition-weighted ecosystem relative change **{synthetic["abundance_only"]["main_ecosystem_relative_change"]:+.2%}**，显示 state 与 abundance-sensitive ecosystem proxy 被区分。
- Receiver autonomous：autonomous response **{synthetic["receiver_autonomous"]["autonomous_response_effect_mean"]:.4f}**，prior explained fraction **{synthetic["receiver_autonomous"]["main_attribution_explained_fraction"]:.4f}**，state effect **{synthetic["receiver_autonomous"]["main_state_effect"]:.6f}**；强 receiver biology 被保留为 residual，而不是强迫归因给 LR prior。
- Ligand only：state effect **{synthetic["ligand_only"]["main_state_effect"]:.4f}**，target response **{synthetic["ligand_only"]["truth_response_effect_mean"]:.4f}**，recovery **{synthetic["ligand_only"]["main_recovery_score"]:.4f}**，低于 active positive control。
- Target only：target response **{synthetic["target_only"]["truth_response_effect_mean"]:.4f}**，state effect **{synthetic["target_only"]["main_state_effect"]:.6f}**，recovery **{synthetic["target_only"]["main_recovery_score"]:.4f}**。
- Receptor knockout：stim receptor availability 和 stim interaction state 均为 **0**，验证 receptor/complex gate。

Synthetic edge AUROC={float(synthetic_truth["auroc"]):.3f}、average precision={float(synthetic_truth["average_precision"]):.3f} 仅在 `synthetic_edge_level_truth_only` 范围内有效。由于只有一个 positive，这两个值不能用于声称概率校准、稳健泛化或真实数据性能；v0.1 formal inference 仍关闭，`p_values=null`、`q_values=null`、`fdr=null`。

![Figure 5](figures/figure05_synthetic.png)

**Figure 5.** A，逐 scenario contract checks。B，主 CXCL10-CXCR3 edge 的 availability-state 与 target response 分离。C，exploratory recovery score；该分数不是 communication probability。D，abundance sensitivity 与 receptor-knockout gate diagnostics。源数据：[figure05_synthetic.csv](source_data/figure05_synthetic.csv)。

## 资源版本

| Resource | Version | Species | Namespace | Adapter | License | Manifest SHA256 prefix |
|---|---|---|---|---|---|---|
{chr(10).join(resource_lines)}

### 数据集资源选择

| Dataset / comparator | LR resource | Role |
|---|---|---|
| Kang IFN-beta / CRYCHIC | CellChatDB human 2.2.0.9001 | CRYCHIC availability 与 canonical run |
| AD skin / CRYCHIC | CellChatDB human 2.2.0.9001 | corrected normalized-only availability 与 canonical run |
| Trophoblast / CRYCHIC | CellPhoneDB human v5.0.0 | HLA-G/TF supportive availability 与 canonical run |
| AD / embryonic archived CellChat | 上游 archived CellChat objects | Reference-only comparator；不是 CRYCHIC output，也不是独立 truth |
| Kang / LIANA | LIANA consensus default | Default-resource rank-only comparator；未与 CRYCHIC harmonize |

LIANA comparator：`{environment["liana_method"]}`，resource=`LIANA consensus default`，seed={environment["liana_seed"]}，jobs={environment["liana_jobs"]}。Archived CellChat reference CSV 的生成输出没有持久化 CellChat R package runtime version；本机当前也未安装 CellChat，因此该 method-version 字段明确记为 `not_recorded`，不能用 CellChatDB 版本替代方法版本。

## 运行与失败状态

| Run | Dataset | Status | Reason code | Result directory | Resource | Seed | Threads |
|---|---|---|---|---|---|---|---|
{chr(10).join(run_lines)}

CRYCHIC 三个 preliminary 输出均未持久化 seed、thread、hardware；这是 provenance gap，已写入 JSON/TSV，而不是补造值。LIANA manifest 记录 seed 和 jobs，但仍未记录硬件。报告生成环境只用于复现绘图：Python {environment["python"]}，pandas {environment["pandas"]}，NumPy {environment["numpy"]}，Matplotlib {environment["matplotlib"]}，CPU logical count {environment["logical_cpus"]}，git revision `{environment["git_revision"]}`。

### Canonical result-schema runner

| Dataset | Status | Wall (s) | Fit/persist (s) | Peak RSS (MiB) | Peak RSS scope | Mode | Persisted table rows | Reason code |
|---|---|---:|---:|---:|---|---|---|---|
{chr(10).join(canonical_lines)}

该表汇总 `benchmark_work/canonical_v01_results` 的 final runner 状态、runtime 与 manifest-declared table rows。Figure 1-2 与 Figure 4 的 trophoblast 部分使用锁定的 preliminary inputs，Figure 3A 和 embryonic panels 是 archived CellChat reference-only；Figure 3B/C 在 corrected AD final interactions 可用时自动切换并重新计算。Final runner 的失败、空表或 mode 状态不会被静默替换成成功。

当前性能解释：{performance_statement}。这些 wall time 与输出行数一起表明 Kang 的 LR top-selection + NicheNet attribution/scoring，以及 AD 的高容量 availability expansion，是当前 v0.1 的主要瓶颈。这是 correctness-first Python baseline，不代表 production scaling，也不能作为跨机器性能结论。

{kang_selection_statement}

Peak RSS 只有在每个 dataset 独立进程重跑并明确记录 `standalone_isolated_process` 时才解释为 standalone peak；顺序复用同一进程的后续 dataset 统一标为 `sequential_process_upper_bound`，避免把 allocator 继承误写为独立峰值。

## 输入校验和

| Workspace-relative path | Role | Bytes | SHA256 prefix |
|---|---|---:|---|
{chr(10).join(input_lines)}

完整 SHA256 同时保存在 [input_manifest.tsv](input_manifest.tsv) 和 [metrics_summary.json](metrics_summary.json)。报告未写入任何机器本地绝对路径。

## 机器可读产物

- [metrics_summary.tsv](metrics_summary.tsv)：一行一个 metric，包含 status、reason_code、method/resource 和解释边界。
- [metrics_summary.json](metrics_summary.json)：包含环境、输入 SHA256、资源版本、run/failure 状态和全部 metric。
- `source_data/figure*.csv`：每张图唯一 source-data 表，`panel` 与图中 panel letter 一致。
- `figures/*.svg`、`*.pdf`、`*.png`：同一绘图对象导出；PNG 为 300 dpi。
- [REPORT.html](REPORT.html)：由 Pandoc 生成的 self-contained HTML，CSS 与图像均内嵌。
- [REPORT.pdf](REPORT.pdf)：由 WeasyPrint 从 self-contained HTML 渲染的 A4 报告。

## 复现命令

在仓库根目录运行：

```bash
uv run --extra resources --extra benchmark --extra plotting python benchmarks/report/generate_canonical_report.py --workspace-root ..
```

`--workspace-root` 必须指向同时包含 `CRYCHIC/`、`benchmark_work/` 与 `databases/` 的 workspace；脚本不硬编码绝对路径。可以用 `--output-dir` 指定相对于仓库根目录的其他输出位置。

生成器会自动完成 Markdown -> self-contained HTML -> A4 PDF。若只需重渲染已有 Markdown，可在 `reports/canonical_v01` 目录运行：

```bash
pandoc REPORT.md --from=gfm --to=html5 --standalone --embed-resources \
  --resource-path=. --css=../../benchmarks/report/report.css \
  --metadata=lang:zh-CN --toc --toc-depth=2 --output=REPORT.html
weasyprint REPORT.html REPORT.pdf
```
"""
    return text


def generate(workspace_root: Path, output_dir: Path, generated_at: str) -> None:
    workspace_root = workspace_root.resolve()
    repo_root = workspace_root / "CRYCHIC"
    benchmark_root = workspace_root / "benchmark_work"
    database_root = workspace_root / "databases"
    for path in (repo_root / "DEVELOPMENT_PLAN.md", benchmark_root, database_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    figures_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_candidates = (
        benchmark_root / "synthetic_v01",
        repo_root / "benchmark_work/synthetic_v01",
    )
    synthetic_root = next(
        (
            candidate
            for candidate in synthetic_candidates
            if (candidate / "summary.json").is_file()
        ),
        None,
    )
    if synthetic_root is None:
        locations = ", ".join(str(path) for path in synthetic_candidates)
        raise FileNotFoundError(
            f"synthetic_v01 summary.json was not found in: {locations}"
        )

    style_path = repo_root / "benchmarks/report/publication.mplstyle"
    plt.style.use(style_path)

    metrics_paths = {
        "kang": benchmark_root / "crychic_kang_prelim/metrics.json",
        "ad": benchmark_root / "crychic_ad_prelim/metrics.json",
        "trophoblast": benchmark_root / "crychic_trophoblast_prelim/metrics.json",
    }
    metrics = {key: _read_json(path) for key, path in metrics_paths.items()}
    with (repo_root / "benchmarks/truth/kang2018_expected_biology.yaml").open(
        encoding="utf-8"
    ) as handle:
        kang_truth = yaml.safe_load(handle)
    with (repo_root / "benchmarks/truth/canonical_supportive_biology.yaml").open(
        encoding="utf-8"
    ) as handle:
        yaml.safe_load(handle)

    embryo_composition = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(
                (benchmark_root / "cellchat_reference").glob(
                    "embryo_*/cell_composition.csv"
                )
            )
        ],
        ignore_index=True,
    )
    book = MetricBook()
    overview, dataset_specs = _overview_data(
        benchmark_root,
        metrics,
        embryo_composition,
        book,
    )
    _save_source(overview, source_dir, "figure01_overview")
    _plot_overview(overview, figures_dir)

    kang_source, kang_summary = _kang_data(benchmark_root, kang_truth, book)
    _save_source(kang_source, source_dir, "figure02_kang")
    _plot_kang(kang_source, figures_dir)

    ad_liana_source, ad_liana_summary = _ad_liana_data(
        benchmark_root,
        database_root,
        repo_root,
        book,
    )
    _save_source(ad_liana_source, source_dir, "figure03_ad_liana")
    _plot_ad_liana(ad_liana_source, figures_dir)

    troph_embryo_source, troph_embryo_summary = _troph_embryo_data(
        benchmark_root,
        database_root,
        book,
    )
    _save_source(troph_embryo_source, source_dir, "figure04_trophoblast_embryo")
    _plot_troph_embryo(troph_embryo_source, figures_dir)

    synthetic_source, synthetic_summary = _synthetic_data(synthetic_root, book)
    _save_source(synthetic_source, source_dir, "figure05_synthetic")
    _plot_synthetic(synthetic_source, figures_dir)

    liana_manifest_path = benchmark_root / "liana_kang/manifest.json"
    liana_manifest = _read_json(liana_manifest_path)
    resources = _resource_records(repo_root)
    canonical_root = benchmark_root / "canonical_v01_results"
    canonical_payload, canonical_records = _canonical_result_records(
        canonical_root,
        workspace_root,
        book,
    )
    runs = _run_records(
        metrics,
        liana_manifest,
        synthetic_summary["summary"],
        _relative(synthetic_root, workspace_root),
    )
    runs.extend(canonical_records)
    for run in runs:
        book.add(
            section="run_status",
            dataset=str(run["dataset"]),
            metric=str(run["run_id"]),
            value_text=str(run["status"]),
            status=str(run["status"]),
            reason_code=str(run["reason_code"]),
            method=str(run["run_id"]),
            resource=str(run["resource"]),
            interpretation="Explicit workflow completion/failure state.",
        )
    for dataset, reason in (
        ("All canonical real data", "real_data_has_no_edge_level_truth"),
        ("AD skin", "normalized_only_and_unpaired_conditions"),
        ("Embryonic skin", "no_subject_replicates"),
        ("Trophoblast", "single_context_only"),
    ):
        book.add(
            section="interpretation_guardrail",
            dataset=dataset,
            metric="formal_edge_inference_available",
            value=0,
            unit="boolean",
            status="not_available",
            reason_code=reason,
            method="report policy",
            interpretation="No p-value, q-value, calibrated probability, or real-data edge AUROC is emitted.",
        )

    input_specs: list[tuple[Path, str]] = []
    input_specs.extend(
        (path, "preliminary run metrics") for path in metrics_paths.values()
    )
    for run_name in (
        "crychic_kang_prelim",
        "crychic_ad_prelim",
        "crychic_trophoblast_prelim",
    ):
        for file_name in (
            "pseudobulk_units.parquet",
            "response_context_means.parquet",
            "response_contrasts.parquet",
            "sample_availability.parquet",
        ):
            path = benchmark_root / run_name / file_name
            if path.is_file():
                input_specs.append((path, "preliminary CRYCHIC table"))
    input_specs.extend(
        [
            (
                repo_root / "benchmarks/report/generate_canonical_report.py",
                "report generator source",
            ),
            (
                repo_root / "benchmarks/report/publication.mplstyle",
                "publication figure style",
            ),
            (
                repo_root / "benchmarks/run_canonical_v01.py",
                "canonical v0.1 runner source",
            ),
            (
                repo_root / "benchmarks/configs/canonical_v01.json",
                "canonical v0.1 benchmark config",
            ),
            (repo_root / "DEVELOPMENT_PLAN.md", "development gate specification"),
            (
                repo_root / "docs/methods/v0.1-exploratory-baseline.md",
                "v0.1 exploratory method specification",
            ),
            (
                benchmark_root / "kang2018_batch2.h5ad",
                "prepared Kang count input for donor-direction reconstruction",
            ),
            (
                benchmark_root / "kang2018_batch2.conversion.json",
                "Kang conversion provenance",
            ),
            (
                benchmark_root / "ad_skin_cellchat.conversion.json",
                "AD conversion provenance",
            ),
            (
                benchmark_root / "trophoblast.conversion.json",
                "trophoblast conversion provenance",
            ),
            (
                benchmark_root / "liana_kang/liana_by_sample.parquet",
                "LIANA default-resource output",
            ),
            (liana_manifest_path, "LIANA run manifest"),
            (
                benchmark_root / "trophoblast/data/active_TFs.tsv",
                "tutorial active-TF supportive labels",
            ),
            (
                benchmark_root / "trophoblast/data/DEGs_inv_trophoblast.tsv",
                "tutorial DEG supportive labels",
            ),
            (
                repo_root / "benchmarks/truth/kang2018_expected_biology.yaml",
                "locked Kang supportive truth",
            ),
            (
                repo_root / "benchmarks/truth/canonical_supportive_biology.yaml",
                "locked canonical supportive truth",
            ),
            (
                database_root / "cellphonedb/v5.0.0/genes.parquet",
                "HLA-G entity mapping",
            ),
            (
                database_root / "cellphonedb/v5.0.0/interactions.parquet",
                "HLA-G interaction mapping",
            ),
            (
                database_root / "cellchat/CELLCHATDB_VERSION.txt",
                "CellChatDB version metadata",
            ),
        ]
    )
    for path in sorted((benchmark_root / "cellchat_reference").glob("*/*.csv")):
        input_specs.append((path, "archived CellChat reference summary"))
    for path in sorted((repo_root / "resources").glob("*.json")):
        input_specs.append((path, "frozen resource manifest"))
    input_specs.append(
        (repo_root / "benchmarks/report/report.css", "report rendering style")
    )
    for file_name in (
        "summary.json",
        "checks.csv",
        "scenario_metrics.csv",
        "edge_metrics.csv",
    ):
        input_specs.append(
            (synthetic_root / file_name, "synthetic v0.1 known-truth benchmark")
        )
    canonical_summary_path = canonical_root / "summary.json"
    if canonical_summary_path.is_file():
        input_specs.append((canonical_summary_path, "canonical v0.1 runner summary"))
    for path in sorted(canonical_root.glob("*/metrics.json")):
        input_specs.append((path, "canonical v0.1 runner metrics"))
    for path in sorted(canonical_root.glob("*/result/run_manifest.json")):
        input_specs.append((path, "canonical persisted result manifest"))
    for path in sorted(canonical_root.glob("*/result/provenance.json")):
        input_specs.append((path, "canonical persisted result provenance"))
    for path in sorted(canonical_root.glob("*/result/interactions.parquet")):
        input_specs.append((path, "canonical interactions used in report diagnostics"))
    kang_final_responses = canonical_root / "kang2018/result/responses.parquet"
    if kang_final_responses.is_file():
        input_specs.append(
            (kang_final_responses, "canonical Kang responses used in final ISG audit")
        )
    final_kang_responses = canonical_root / "kang2018/result/responses.parquet"
    if final_kang_responses.is_file():
        input_specs.append(
            (final_kang_responses, "canonical Kang responses used in final ISG audit")
        )
    seen: set[Path] = set()
    inputs: list[dict[str, Any]] = []
    for path, role in input_specs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        inputs.append(_input_record(path, workspace_root, role))
    inputs.sort(key=lambda item: str(item["path"]))
    pd.DataFrame(inputs).to_csv(
        output_dir / "input_manifest.tsv", sep="\t", index=False
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "crychic": _package_version("CRYCHIC"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "anndata": _package_version("anndata"),
        "matplotlib": _package_version("matplotlib"),
        "pyarrow": _package_version("pyarrow"),
        "git_revision": _git_revision(repo_root),
        "report_randomness": "none",
        "liana_method": f"{liana_manifest['method']} {liana_manifest['method_version']}",
        "liana_seed": liana_manifest["seed"],
        "liana_jobs": liana_manifest["n_jobs"],
        "cellchat_reference_method_version": "not_recorded",
        "pandoc": _tool_version("pandoc", "--version"),
        "weasyprint": _tool_version("weasyprint", "--version"),
    }
    metric_frame = book.frame()
    metric_frame.to_csv(
        output_dir / "metrics_summary.tsv", sep="\t", index=False, na_rep=""
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "generated_at": generated_at,
        "scope": "canonical real-data preliminary availability and supportive biology",
        "statistical_guardrails": {
            "real_data_edge_truth_available": False,
            "edge_auroc_reported": False,
            "p_values_reported": False,
            "q_values_reported": False,
            "calibrated_probabilities_reported": False,
            "cellchat_scores_treated_as_truth_probability": False,
            "liana_default_resource_harmonized_with_crychic": False,
            "synthetic_edge_truth_scope": "single-positive smoke only",
        },
        "environment": environment,
        "resources": resources,
        "runs": runs,
        "canonical_results": canonical_payload,
        "synthetic_summary": {
            key: _native(value) for key, value in synthetic_summary["summary"].items()
        },
        "inputs": inputs,
        "metrics": [
            {key: _native(value) for key, value in row.items()}
            for row in metric_frame.to_dict("records")
        ],
        "figures": [
            {
                "id": stem,
                "source_data": f"source_data/{stem}.csv",
                "formats": [
                    f"figures/{stem}.svg",
                    f"figures/{stem}.pdf",
                    f"figures/{stem}.png",
                ],
                "png_dpi": 300,
            }
            for stem in (
                "figure01_overview",
                "figure02_kang",
                "figure03_ad_liana",
                "figure04_trophoblast_embryo",
                "figure05_synthetic",
            )
        ],
        "report_artifacts": ["REPORT.md", "REPORT.html", "REPORT.pdf"],
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = _report_markdown(
        generated_at=generated_at,
        environment=environment,
        dataset_specs=dataset_specs,
        resources=resources,
        runs=runs,
        inputs=inputs,
        kang=kang_summary,
        ad_liana=ad_liana_summary,
        troph_embryo=troph_embryo_summary,
        synthetic=synthetic_summary,
        canonical=canonical_payload,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    _render_report(output_dir, repo_root / "benchmarks/report/report.css")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        required=True,
        type=Path,
        help="Directory containing CRYCHIC, benchmark_work, and databases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/canonical_v01"),
        help="Absolute path or path relative to the CRYCHIC repository root",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="ISO-8601 report timestamp; defaults to the current local time",
    )
    args = parser.parse_args()
    generated_at = args.generated_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    generate(args.workspace_root, args.output_dir, generated_at)


if __name__ == "__main__":
    main()
