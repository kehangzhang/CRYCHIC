"""Render a checksum-bound report from a unified spatial DES summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_SCHEMA = "crychic-spatial-des-benchmark-summary-v1"
REPORT_SCHEMA = "crychic-spatial-des-report-v1"
LEADERBOARD_FILENAME = "spatial_des_leaderboard.tsv"
SUMMARY_SOURCE_FILENAME = "spatial_des_benchmark_summary.tsv"
SUMMARY_MANIFEST_SOURCE_FILENAME = "spatial_des_benchmark_summary_manifest.json"
PUBLICATION_FILENAMES = frozenset(
    {
        "README.md",
        "manifest.json",
        SUMMARY_SOURCE_FILENAME,
        SUMMARY_MANIFEST_SOURCE_FILENAME,
        LEADERBOARD_FILENAME,
        "figure_des_median.png",
        "figure_des_mean.png",
        "figure_rank_coverage.png",
    }
)
MAX_PUBLICATION_FILE_BYTES = 25 * 1024 * 1024
MAX_PUBLICATION_BUNDLE_BYTES = 50 * 1024 * 1024
FORBIDDEN_PUBLICATION_SUFFIXES = frozenset(
    {".h5", ".h5ad", ".log", ".out", ".parquet", ".rda", ".rds", ".rdata"}
)
POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?!/)"
    r"(?:[A-Za-z0-9._~+-]+/)*[A-Za-z0-9._~+-]+"
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")

METHOD_COLORS = {
    "crychic": "#0072B2",
    "scseqcommdiff": "#D55E00",
    "scdiffcom": "#009E73",
    "cellchat": "#E69F00",
    "cellchat_condition_aware": "#E69F00",
    "liana_rank_aggregate": "#CC79A7",
    "liana_plus": "#CC79A7",
    "liana_plus_de": "#CC79A7",
    "multinichenet": "#56B4E9",
}
PRIMARY_REPORT_EXCLUDED_METHODS = frozenset(
    {"liana_rank_aggregate", "multinichenet"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_tsv_atomic(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        table.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_publication_bundle(root: Path) -> None:
    """Validate the exact, portable Git publication artifact set."""

    if not root.is_dir():
        raise ValueError(f"publication bundle does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        if path.suffix.lower() in FORBIDDEN_PUBLICATION_SUFFIXES:
            raise ValueError(f"forbidden publication artifact type: {path.name}")
    relative_names = {path.relative_to(root).as_posix() for path in paths}
    if relative_names != PUBLICATION_FILENAMES:
        missing = sorted(PUBLICATION_FILENAMES.difference(relative_names))
        unexpected = sorted(relative_names.difference(PUBLICATION_FILENAMES))
        raise ValueError(
            f"publication artifact set mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    total_bytes = 0
    for path in paths:
        size = path.stat().st_size
        total_bytes += size
        if size > MAX_PUBLICATION_FILE_BYTES:
            raise ValueError(f"publication artifact exceeds size limit: {path.name}")
        if path.suffix.lower() == ".png":
            if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"invalid PNG publication artifact: {path.name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"publication artifact is not UTF-8 text: {path.name}"
            ) from error
        if POSIX_ABSOLUTE_PATH.search(text) or WINDOWS_ABSOLUTE_PATH.search(text):
            raise ValueError(
                f"absolute machine path found in publication artifact: {path.name}"
            )
    if total_bytes > MAX_PUBLICATION_BUNDLE_BYTES:
        raise ValueError("publication bundle exceeds total size limit")

    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("publication manifest is invalid JSON") from error
    valid_schema = (
        isinstance(manifest, dict) and manifest.get("schema_version") == REPORT_SCHEMA
    )
    if not valid_schema:
        raise ValueError("unsupported publication manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("publication manifest is not complete")
    output_records = manifest.get("outputs")
    expected_outputs = PUBLICATION_FILENAMES.difference({"manifest.json"})
    if not isinstance(output_records, dict) or set(output_records) != expected_outputs:
        raise ValueError("publication manifest output inventory is incomplete")
    for filename, record in output_records.items():
        path = root / filename
        if not isinstance(record, dict):
            raise ValueError(f"invalid publication output record: {filename}")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"publication byte count mismatch: {filename}")
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"publication SHA256 mismatch: {filename}")

    summary_path = root / SUMMARY_SOURCE_FILENAME
    summary_manifest_path = root / SUMMARY_MANIFEST_SOURCE_FILENAME
    summary_manifest = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
    if summary_manifest.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError("unsupported copied summary manifest schema")
    summary_output = summary_manifest.get("output")
    if not isinstance(summary_output, dict):
        raise ValueError("copied summary manifest lacks output metadata")
    if summary_output.get("sha256") != _sha256(summary_path):
        raise ValueError("copied summary SHA256 disagrees with its manifest")
    publication_input = manifest.get("input")
    expected_input = {
        "summary_filename": SUMMARY_SOURCE_FILENAME,
        "summary_sha256": _sha256(summary_path),
        "summary_manifest_filename": SUMMARY_MANIFEST_SOURCE_FILENAME,
        "summary_manifest_sha256": _sha256(summary_manifest_path),
    }
    if publication_input != expected_input:
        raise ValueError("publication input binding is inconsistent")


def _read_summary(summary_path: Path, manifest_path: Path) -> pd.DataFrame:
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("summary table and manifest must both exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("summary manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("summary manifest must contain an object")
    if manifest.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError("unsupported spatial DES summary schema")
    if manifest.get("status") != "complete":
        raise ValueError("spatial DES summary is not complete")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("summary manifest lacks output metadata")
    if output.get("sha256") != _sha256(summary_path):
        raise ValueError("summary SHA256 disagrees with manifest")

    table = pd.read_csv(summary_path, sep="\t", low_memory=False)
    required = {
        "comparison_panel_id",
        "dataset",
        "scenario",
        "analysis_unit",
        "expected_variant",
        "method",
        "method_version",
        "resource",
        "ranking_semantics",
        "des_strata_expected",
        "des_strata_observed",
        "des_median",
        "des_mean",
        "rank_eligible",
        "median_rank",
        "mean_rank",
        "rank_eligible_fraction_min",
        "rank_eligible_fraction_mean",
        "expected_set_coverage_min",
        "expected_set_coverage_mean",
    }
    missing = required.difference(table.columns)
    if missing or table.empty:
        raise ValueError(f"summary is empty or missing columns: {sorted(missing)}")
    if output.get("rows") != len(table):
        raise ValueError("summary row count disagrees with manifest")
    identity = [
        "comparison_panel_id",
        "method",
        "method_version",
        "resource",
        "ranking_semantics",
    ]
    if table.duplicated(identity).any():
        raise ValueError("summary contains duplicate method identities")
    excluded = set(table["method"].astype(str)).intersection(
        PRIMARY_REPORT_EXCLUDED_METHODS
    )
    if excluded:
        raise ValueError(
            "primary report contains declared sensitivity or skipped methods: "
            f"{sorted(excluded)}"
        )

    eligible = table["rank_eligible"].map(
        lambda value: value
        if isinstance(value, bool)
        else {"True": True, "False": False}.get(str(value))
    )
    if eligible.isna().any():
        raise ValueError("rank_eligible must contain booleans")
    table["rank_eligible"] = eligible.astype(bool)
    ranks = pd.to_numeric(table["median_rank"], errors="coerce")
    if ranks.loc[table["rank_eligible"]].isna().any():
        raise ValueError("eligible rows require a rank")
    if ranks.loc[~table["rank_eligible"]].notna().any():
        raise ValueError("ineligible rows must not have a rank")
    table["median_rank"] = ranks.astype("Int64")
    mean_ranks = pd.to_numeric(table["mean_rank"], errors="coerce")
    if mean_ranks.loc[table["rank_eligible"]].isna().any():
        raise ValueError("eligible rows require a mean-DES rank")
    if mean_ranks.loc[~table["rank_eligible"]].notna().any():
        raise ValueError("ineligible rows must not have a mean-DES rank")
    table["mean_rank"] = mean_ranks.astype("Int64")
    for panel_id, group in table.groupby("comparison_panel_id", observed=True):
        observed = group.loc[group["rank_eligible"]]
        expected = observed["des_median"].rank(method="min", ascending=False)
        supplied = observed["median_rank"].astype(float)
        if not np.array_equal(expected.to_numpy(), supplied.to_numpy()):
            raise ValueError(f"rank mismatch in comparison panel {panel_id}")
        expected_mean = observed["des_mean"].rank(method="min", ascending=False)
        supplied_mean = observed["mean_rank"].astype(float)
        if not np.array_equal(expected_mean.to_numpy(), supplied_mean.to_numpy()):
            raise ValueError(f"mean rank mismatch in comparison panel {panel_id}")
    return table


def _display_method(method: str) -> str:
    labels = {
        "crychic": "CRYCHIC",
        "scseqcommdiff": "scSeqCommDiff",
        "scdiffcom": "scDiffCom",
        "cellchat": "CellChat",
        "cellchat_condition_aware": "CellChat",
        "liana_rank_aggregate": "LIANA rank aggregate",
        "liana_plus": "LIANA+",
        "liana_plus_de": "LIANA+",
        "multinichenet": "MultiNicheNet",
    }
    return labels.get(method, method)


def _panel_label(group: pd.DataFrame) -> str:
    first = group.iloc[0]
    dataset_id = str(first["dataset"])
    dataset = {
        "Kuppe_MI_spatial_CTRL_vs_IZ": "Kuppe MI: CTRL vs IZ",
        "lerma_martin_ms_ctrl_vs_chronic_active": (
            "Lerma-Martin MS: control vs chronic active"
        ),
    }.get(dataset_id, dataset_id.replace("_", " "))
    scenario = str(first["scenario"]).replace("_", "-")
    unit_id = str(first["analysis_unit"])
    unit = {
        "condition_level": "condition level",
        "sample_id": "sample",
        "subject_id": "subject",
    }.get(unit_id, unit_id.replace("_", " "))
    variant = str(first["expected_variant"]).replace("_", " ")
    return f"{dataset}\n{scenario} | {unit} | {variant}"


def _leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison_panel_id",
        "dataset",
        "scenario",
        "analysis_unit",
        "expected_variant",
        "method",
        "method_version",
        "resource",
        "ranking_semantics",
        "rank_eligible",
        "median_rank",
        "mean_rank",
        "des_median",
        "des_mean",
        "des_strata_observed",
        "des_strata_expected",
        "rank_eligible_fraction_min",
        "rank_eligible_fraction_mean",
        "expected_set_coverage_min",
        "expected_set_coverage_mean",
    ]
    result = summary.loc[:, columns].copy()
    method_index = list(result.columns).index("method")
    result.insert(
        method_index + 1,
        "method_display",
        result["method"].map(_display_method),
    )
    return result.sort_values(
        [
            "dataset",
            "scenario",
            "analysis_unit",
            "expected_variant",
            "comparison_panel_id",
            "median_rank",
            "method",
        ],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )


def _plot_facets(
    table: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    x_label: str,
    x_limit: tuple[float, float] | None = None,
) -> None:
    # The leaderboard is already sorted dataset-first and scenario-second.
    panels = list(table.groupby("comparison_panel_id", sort=False, observed=True))
    columns = min(2, len(panels))
    rows = math.ceil(len(panels) / columns)
    max_methods = max(len(group) for _, group in panels)
    panel_height = max(3.6, 1.8 + 0.62 * max_methods)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.4 * columns, panel_height * rows),
        squeeze=False,
    )
    flat_axes = tuple(axes.flat)
    for axis, (_, group) in zip(flat_axes, panels, strict=False):
        ordered = group.sort_values(
            ["rank_eligible", metric, "method"],
            ascending=[True, True, True],
            kind="stable",
        )
        labels = ordered["method"].map(_display_method)
        values = pd.to_numeric(ordered[metric], errors="coerce")
        colors = [
            METHOD_COLORS.get(str(method), "#777777")
            if eligible
            else "#BDBDBD"
            for method, eligible in zip(
                ordered["method"], ordered["rank_eligible"], strict=True
            )
        ]
        positions = np.arange(len(ordered))
        axis.barh(positions, values.fillna(0.0), color=colors, height=0.68)
        axis.set_yticks(positions, labels)
        axis.set_xlabel(x_label)
        axis.set_title(_panel_label(ordered), fontsize=9)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
        if x_limit is not None:
            axis.set_xlim(*x_limit)
        for position, (_, row) in enumerate(ordered.iterrows()):
            value = row[metric]
            if pd.isna(value):
                label = "not ranked"
                x = 0.01
            elif metric in {"des_median", "des_mean"} and bool(
                row["rank_eligible"]
            ):
                rank_column = "median_rank" if metric == "des_median" else "mean_rank"
                label = f"{float(value):.3f}  rank {int(row[rank_column])}"
                x = float(value)
            else:
                label = f"{float(value):.3f}"
                x = float(value)
            axis.annotate(
                label,
                (x, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
    for axis in flat_axes[len(panels) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "CRYCHIC spatial DES report"},
    )
    plt.close(figure)


def _markdown_table(table: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "scenario",
        "analysis_unit",
        "expected_variant",
        "comparison_panel_id",
        "method_display",
        "median_rank",
        "mean_rank",
        "des_median",
        "des_mean",
        "rank_eligible_fraction_min",
        "expected_set_coverage_min",
    ]
    header = [
        "Dataset",
        "Scenario",
        "Unit",
        "Truth variant",
        "Panel",
        "Method",
        "Median rank",
        "Mean rank",
        "Median DES",
        "Mean DES",
        "Min rank coverage",
        "Min truth coverage",
    ]
    rows = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for record in table.loc[:, columns].itertuples(index=False, name=None):
        formatted = [str(value) for value in record[:6]]
        for value in record[6:8]:
            formatted.append("not ranked" if pd.isna(value) else str(int(value)))
        for value in record[8:]:
            formatted.append("NA" if pd.isna(value) else f"{float(value):.3f}")
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def run(
    summary_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    outputs = [
        output_dir / SUMMARY_SOURCE_FILENAME,
        output_dir / SUMMARY_MANIFEST_SOURCE_FILENAME,
        output_dir / LEADERBOARD_FILENAME,
        output_dir / "figure_des_median.png",
        output_dir / "figure_des_mean.png",
        output_dir / "figure_rank_coverage.png",
        output_dir / "README.md",
        output_dir / "manifest.json",
    ]
    conflicts = [path.name for path in outputs if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {conflicts}")
    summary = _read_summary(summary_path.resolve(), manifest_path.resolve())
    leaderboard = _leaderboard(summary)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = output_dir / LEADERBOARD_FILENAME
    summary_source_path = output_dir / SUMMARY_SOURCE_FILENAME
    summary_manifest_source_path = output_dir / SUMMARY_MANIFEST_SOURCE_FILENAME
    des_figure = output_dir / "figure_des_median.png"
    mean_des_figure = output_dir / "figure_des_mean.png"
    coverage_figure = output_dir / "figure_rank_coverage.png"
    readme_path = output_dir / "README.md"
    _write_bytes_atomic(summary_source_path, summary_path.read_bytes())
    _write_bytes_atomic(
        summary_manifest_source_path,
        manifest_path.read_bytes(),
    )
    _write_tsv_atomic(leaderboard_path, leaderboard)
    _plot_facets(
        leaderboard,
        des_figure,
        metric="des_median",
        x_label="Median spatial DES (higher is better)",
    )
    _plot_facets(
        leaderboard,
        mean_des_figure,
        metric="des_mean",
        x_label="Mean spatial DES (secondary; higher is better)",
    )
    _plot_facets(
        leaderboard,
        coverage_figure,
        metric="rank_eligible_fraction_min",
        x_label="Minimum eligible pair coverage",
        x_limit=(0.0, 1.08),
    )
    report = (
        "# Spatial differential CCC benchmark\n\n"
        "Results are ranked only within checksum-compatible comparison panels. "
        "Panels never mix datasets, spatial truth variants, analysis units, self-"
        "pair policies, tie policies, or DES semantics. A method is ranked only "
        "when all eight condition-by-top-"
        "fraction strata are observed. The primary report excludes the LIANA "
        "1.7.3 descriptive sensitivity arm. MultiNicheNet is a declared skip "
        "because no validated frozen adapter was available; see "
        "the benchmark protocol for the full scope and deviations.\n\n"
        "## Leaderboard\n\n"
        f"{_markdown_table(leaderboard)}\n\n"
        "## Figures\n\n"
        "![Median DES leaderboard](figure_des_median.png)\n\n"
        "![Mean DES leaderboard](figure_des_mean.png)\n\n"
        "![Eligible pair coverage](figure_rank_coverage.png)\n"
    )
    _write_text_atomic(readme_path, report)

    output_records: dict[str, dict[str, object]] = {}
    for path in outputs[:-1]:
        output_records[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "input": {
            "summary_filename": summary_source_path.name,
            "summary_sha256": _sha256(summary_source_path),
            "summary_manifest_filename": summary_manifest_source_path.name,
            "summary_manifest_sha256": _sha256(summary_manifest_source_path),
        },
        "protocol": {
            "no_cross_panel_ranking": True,
            "rank_metric": "median spatial DES descending",
            "secondary_rank_metric": "mean spatial DES descending",
            "complete_strata_required": 8,
        },
        "outputs": output_records,
    }
    manifest_output = output_dir / "manifest.json"
    _write_text_atomic(
        manifest_output,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    validate_publication_bundle(output_dir)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--summary-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        args.summary,
        args.summary_manifest,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
