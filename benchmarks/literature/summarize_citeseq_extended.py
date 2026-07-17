"""Audit and summarize the seven-dataset CITE-seq benchmark extension."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json

REQUIRED_EVALUATION_FILES = (
    "point_estimates.tsv",
    "bootstrap_summary.tsv",
    "negative_sampling_summary.tsv",
    "evaluated_scores.parquet",
)
MOUSE_EXCLUDED_METHODS = (
    "CellPhoneDB composite",
    "CellPhoneDB p-value",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _artifact(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(workspace.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _read_truth(path: Path) -> tuple[pd.DataFrame, str]:
    table = pd.read_csv(path, sep="\t")
    receptor_column = "receptor" if "receptor" in table.columns else "receptor_gene"
    required = {"receiver", receptor_column, "is_positive"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"truth table {path} lacks {sorted(missing)}")
    return table, receptor_column


def _resource_overlap(
    truth: pd.DataFrame, receptor_column: str, resource_path: Path
) -> tuple[int, int, int]:
    resource = pd.read_csv(resource_path, sep="\t")
    if not {"ligand", "receptor"}.issubset(resource.columns):
        raise ValueError(f"resource lacks ligand/receptor: {resource_path}")
    truth_receptors = set(truth[receptor_column].astype(str).str.upper())
    resource_receptors = set(resource["receptor"].astype(str).str.upper())
    overlap = truth_receptors.intersection(resource_receptors)
    covered_pairs = resource.loc[
        resource["receptor"].astype(str).str.upper().isin(overlap),
        ["ligand", "receptor"],
    ].drop_duplicates()
    return len(resource), len(overlap), len(covered_pairs)


def _evaluation_dir(result_root: Path, arm: str) -> Path:
    if arm == "H-common/resource_fixed":
        return result_root / "H-common" / "evaluation_resource_fixed"
    if arm == "H-common/independent":
        return result_root / "H-common" / "evaluation_independent"
    if arm == "native/independent":
        return result_root / "native" / "evaluation_independent"
    raise ValueError(f"unknown arm: {arm}")


def _write_evaluation_manifest(
    *,
    workspace: Path,
    dataset: dict[str, Any],
    arm: str,
    evaluation_dir: Path,
    truth_path: Path,
    result_root: Path,
) -> Path:
    for filename in REQUIRED_EVALUATION_FILES:
        if not (evaluation_dir / filename).is_file():
            raise FileNotFoundError(evaluation_dir / filename)
    point = pd.read_csv(evaluation_dir / "point_estimates.tsv", sep="\t")
    methods = sorted(point["method"].astype(str).unique())
    if dataset["species"] == "mouse":
        unexpected = sorted(set(methods).intersection(MOUSE_EXCLUDED_METHODS))
        if unexpected:
            raise ValueError(
                f"unsupported mouse CellPhoneDB components present: {unexpected}"
            )

    input_paths = [truth_path]
    if arm.startswith("H-common"):
        input_paths.extend(
            [
                result_root
                / "H-common"
                / "liana_cellchat_literature"
                / "manifest.json",
                result_root / "H-common" / "crychic_availability" / "manifest.json",
            ]
        )
    else:
        input_paths.extend(
            [
                result_root / "native" / "liana_cellchat_literature" / "manifest.json",
                result_root
                / dataset["native_crychic_dir"]
                / "crychic_availability"
                / "manifest.json",
            ]
        )
    if arm == "H-common/resource_fixed":
        input_paths.extend(
            [
                _resolve(workspace, dataset["hcommon_resource"]),
                _resolve(workspace, dataset["hcommon_resource_manifest"]),
            ]
        )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "schema_version": "crychic-citeseq-evaluation-bundle-v1",
        "status": "complete",
        "dataset": dataset["dataset_id"],
        "species": dataset["species"],
        "comparison_arm": arm,
        "methods": methods,
        "excluded_methods": (
            list(MOUSE_EXCLUDED_METHODS) if dataset["species"] == "mouse" else []
        ),
        "exclusion_reason": (
            "CellPhoneDB mouse is unavailable; LIANA placeholder component columns "
            "were not treated as executed methods"
            if dataset["species"] == "mouse"
            else None
        ),
        "inputs": [_artifact(path, workspace) for path in input_paths],
        "outputs": [
            _artifact(evaluation_dir / filename, workspace)
            for filename in REQUIRED_EVALUATION_FILES
        ],
    }
    path = evaluation_dir / "manifest.json"
    write_json(path, manifest)
    return path


def _write_not_evaluable_manifests(
    *,
    workspace: Path,
    dataset: dict[str, Any],
    result_root: Path,
    truth_path: Path,
    resource_path: Path,
    resource_manifest_path: Path,
    truth_receptors: int,
    resource_receptors_covered: int,
) -> list[Path]:
    paths: list[Path] = []
    for mode in ("resource_fixed", "independent"):
        directory = result_root / "H-common" / f"evaluation_{mode}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "crychic-citeseq-evaluation-bundle-v1",
            "status": "not_evaluable",
            "reason_code": dataset["hcommon_reason"],
            "reason": (
                "No receptor in the observed ADT truth overlaps the frozen "
                "human H-common resource; no score is reported."
            ),
            "dataset": dataset["dataset_id"],
            "species": dataset["species"],
            "comparison_arm": f"H-common/{mode}",
            "evidence": {
                "truth_receptors": truth_receptors,
                "resource_receptors_covered": resource_receptors_covered,
            },
            "inputs": [
                _artifact(truth_path, workspace),
                _artifact(resource_path, workspace),
                _artifact(resource_manifest_path, workspace),
            ],
            "outputs": [],
        }
        path = directory / "manifest.json"
        write_json(path, manifest)
        paths.append(path)
    return paths


def _parse_time_file(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    memory = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    exit_status = re.search(r"Exit status:\s*(\d+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time [^\n]*\):\s*([^\n]+)", text)
    if not memory and not exit_status and not elapsed:
        return None
    return {
        "time_file": str(path),
        "peak_rss_kb": int(memory.group(1)) if memory else pd.NA,
        "elapsed": elapsed.group(1).strip() if elapsed else "",
        "exit_status": int(exit_status.group(1)) if exit_status else pd.NA,
    }


def _git_state(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def _markdown_table(table: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _build_report(
    inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    skipped: pd.DataFrame,
    max_peak_rss_kb: int,
    memory_total_kb: int,
) -> str:
    new_datasets = metrics.loc[
        metrics["short_name"].isin(["cbmc", "sln_111", "sln_208"])
    ]
    primary = new_datasets.loc[
        new_datasets["comparison_arm"] == "H-common/resource_fixed",
        ["dataset", "species", "method", "auroc", "average_precision"],
    ].sort_values(["dataset", "auroc"], ascending=[True, False])
    native = new_datasets.loc[
        new_datasets["comparison_arm"] == "native/independent",
        ["dataset", "species", "method", "auroc", "average_precision"],
    ].sort_values(["dataset", "auroc"], ascending=[True, False])
    primary_top = primary.groupby("dataset", sort=False).head(3)
    native_top = native.groupby("dataset", sort=False).head(3)

    lines = [
        "# Extended seven-dataset CITE-seq benchmark",
        "",
        "## Scope",
        "",
        (
            f"The frozen inventory contains {len(inventory)} public CITE-seq datasets "
            f"({(inventory['species'] == 'human').sum()} human, "
            f"{(inventory['species'] == 'mouse').sum()} mouse). CBMC, SLN111 and "
            "SLN208 are the newly completed datasets. ADT was held out from all "
            "communication methods and used only for receiver-receptor truth."
        ),
        "",
        "The same-resource `H-common/resource_fixed` arm is the only direct method "
        "comparison within a dataset/species resource. Native independent results "
        "are descriptive because LIANA and CRYCHIC use different native resources.",
        "",
        "## Newly completed results",
        "",
        "Top H-common/resource-fixed AUROC rows (CBMC is not evaluable):",
        "",
        _markdown_table(primary_top),
        "",
        "Top native/independent AUROC rows:",
        "",
        _markdown_table(native_top),
        "",
        "## Skips and non-evaluable arms",
        "",
        _markdown_table(skipped),
        "",
        "## Audit limits",
        "",
        "- Mouse CellPhoneDB is unavailable and was skipped. LIANA's placeholder "
        "CellPhoneDB component columns were explicitly excluded from mouse metrics.",
        "- CBMC has 14 observed receptor truth genes, but none overlap the frozen "
        "638-pair human H-common resource; reporting a metric would be invalid.",
        "- All seven inputs have one sample, one subject and one context. They cannot "
        "estimate condition-specific or differential communication.",
        "- ADT validates receiver-receptor specificity, not ligand binding, sender "
        "causality or downstream signaling activation.",
        "- Aggregate means in `method_summary.tsv` are descriptive and species "
        "stratified; they are not pooled inferential estimates.",
        "",
        "## Resource and memory audit",
        "",
        (
            f"The highest measured peak RSS was {max_peak_rss_kb / 1024**2:.2f} GiB "
            f"({100 * max_peak_rss_kb / memory_total_kb:.2f}% of host RAM), below "
            "the requested 70% ceiling. Runtime records and checksums are in the TSV "
            "and JSON manifests."
        ),
        "",
        "The human H-common resource contains 638 frozen simple LR pairs. The mouse "
        "H-common resource contains 705 LIANA-consensus x CellChatDB-mouse simple "
        "pairs. Native mouse LIANA uses the checksum-pinned 4,015-pair MGI resource.",
        "",
    ]
    return "\n".join(lines)


def summarize(
    *,
    workspace: Path,
    dataset_manifest: Path,
    output_dir: Path,
    memory_cap_fraction: float,
    overwrite: bool,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    dataset_manifest = dataset_manifest.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _read_json(dataset_manifest)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 7:
        raise ValueError("exactly seven explicit CITE-seq datasets are required")
    result_base = workspace / "benchmark_work/literature_reproduction/results/citeseq"
    inventory_records: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    skip_records: list[dict[str, Any]] = []
    evaluation_manifests: list[Path] = []

    for dataset in datasets:
        short_name = str(dataset["short_name"])
        result_root = result_base / short_name
        h5ad_path = _resolve(workspace, dataset["prepared_h5ad"])
        truth_path = _resolve(workspace, dataset["truth"])
        resource_path = _resolve(workspace, dataset["hcommon_resource"])
        resource_manifest_path = _resolve(
            workspace, dataset["hcommon_resource_manifest"]
        )
        for path in (h5ad_path, truth_path, resource_path, resource_manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        truth, receptor_column = _read_truth(truth_path)
        resource_pairs, covered_receptors, covered_pairs = _resource_overlap(
            truth, receptor_column, resource_path
        )
        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            required_obs = {"sample_id", "subject_id", "cell_type", "context"}
            if not required_obs.issubset(adata.obs.columns):
                raise ValueError(
                    f"prepared input lacks frozen obs columns: {h5ad_path}"
                )
            shape = tuple(adata.shape)
            obs_counts = {
                column: int(adata.obs[column].astype(str).nunique())
                for column in required_obs
            }
        finally:
            adata.file.close()

        inventory_records.append(
            {
                "short_name": short_name,
                "dataset": dataset["dataset_id"],
                "species": dataset["species"],
                "source": dataset["source"],
                "cells": shape[0],
                "genes": shape[1],
                "cell_types": obs_counts["cell_type"],
                "samples": obs_counts["sample_id"],
                "subjects": obs_counts["subject_id"],
                "contexts": obs_counts["context"],
                "truth_rows": len(truth),
                "truth_positive_rows": int(
                    pd.to_numeric(truth["is_positive"], errors="raise").sum()
                ),
                "truth_receptors": int(
                    truth[receptor_column].astype(str).str.upper().nunique()
                ),
                "hcommon_resource_pairs": resource_pairs,
                "hcommon_truth_receptors_covered": covered_receptors,
                "hcommon_truth_lr_pairs_covered": covered_pairs,
                "hcommon_status": dataset["hcommon_status"],
                "h5ad_sha256": sha256_file(h5ad_path),
                "truth_sha256": sha256_file(truth_path),
                "hcommon_resource_sha256": sha256_file(resource_path),
                "differential_inference_estimable": False,
            }
        )

        if dataset["hcommon_status"] == "complete":
            if covered_receptors == 0:
                raise ValueError(
                    "complete H-common arm has zero truth coverage: "
                    f"{short_name}"
                )
            for arm in ("H-common/resource_fixed", "H-common/independent"):
                directory = _evaluation_dir(result_root, arm)
                evaluation_manifests.append(
                    _write_evaluation_manifest(
                        workspace=workspace,
                        dataset=dataset,
                        arm=arm,
                        evaluation_dir=directory,
                        truth_path=truth_path,
                        result_root=result_root,
                    )
                )
                table = pd.read_csv(directory / "point_estimates.tsv", sep="\t")
                table["short_name"] = short_name
                table["species"] = dataset["species"]
                table["comparison_arm"] = arm
                metric_frames.append(table)
        else:
            if covered_receptors != 0:
                raise ValueError(
                    "not-evaluable H-common arm unexpectedly has coverage: "
                    f"{short_name}"
                )
            evaluation_manifests.extend(
                _write_not_evaluable_manifests(
                    workspace=workspace,
                    dataset=dataset,
                    result_root=result_root,
                    truth_path=truth_path,
                    resource_path=resource_path,
                    resource_manifest_path=resource_manifest_path,
                    truth_receptors=inventory_records[-1]["truth_receptors"],
                    resource_receptors_covered=covered_receptors,
                )
            )
            for arm in ("H-common/resource_fixed", "H-common/independent"):
                skip_records.append(
                    {
                        "dataset": dataset["dataset_id"],
                        "species": dataset["species"],
                        "comparison_arm": arm,
                        "method": "all",
                        "status": "not_evaluable",
                        "reason_code": dataset["hcommon_reason"],
                    }
                )

        native_arm = "native/independent"
        native_dir = _evaluation_dir(result_root, native_arm)
        evaluation_manifests.append(
            _write_evaluation_manifest(
                workspace=workspace,
                dataset=dataset,
                arm=native_arm,
                evaluation_dir=native_dir,
                truth_path=truth_path,
                result_root=result_root,
            )
        )
        native = pd.read_csv(native_dir / "point_estimates.tsv", sep="\t")
        native["short_name"] = short_name
        native["species"] = dataset["species"]
        native["comparison_arm"] = native_arm
        metric_frames.append(native)
        if dataset["species"] == "mouse":
            for method in MOUSE_EXCLUDED_METHODS:
                for arm in (
                    "H-common/resource_fixed",
                    "H-common/independent",
                    native_arm,
                ):
                    skip_records.append(
                        {
                            "dataset": dataset["dataset_id"],
                            "species": "mouse",
                            "comparison_arm": arm,
                            "method": method,
                            "status": "skipped",
                            "reason_code": "cellphonedb_mouse_unavailable",
                        }
                    )

    inventory = pd.DataFrame.from_records(inventory_records)
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    skipped = pd.DataFrame.from_records(skip_records).drop_duplicates()
    summary = (
        metrics.groupby(["comparison_arm", "species", "method"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_auroc=("auroc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            min_auroc=("auroc", "min"),
            max_auroc=("auroc", "max"),
        )
        .sort_values(
            ["comparison_arm", "species", "mean_auroc"],
            ascending=[True, True, False],
        )
    )

    time_records: list[dict[str, Any]] = []
    time_roots = [
        workspace
        / "benchmark_work/literature_reproduction/citeseq/prepared/cbmc_seuratdata",
        workspace / "benchmark_work/literature_reproduction/citeseq/prepared/sln_111",
        workspace / "benchmark_work/literature_reproduction/citeseq/prepared/sln_208",
        result_base / "cbmc",
        result_base / "sln_111",
        result_base / "sln_208",
    ]
    for root in time_roots:
        for path in sorted(root.rglob("*time.txt")):
            record = _parse_time_file(path)
            if record is not None:
                record["time_file"] = str(path.relative_to(workspace))
                time_records.append(record)
    runtime = pd.DataFrame.from_records(time_records)
    max_peak_rss_kb = int(pd.to_numeric(runtime["peak_rss_kb"]).max())
    memory_total_kb = int(
        re.search(
            r"MemTotal:\s*(\d+) kB",
            Path("/proc/meminfo").read_text(encoding="utf-8"),
        ).group(1)
    )
    if max_peak_rss_kb / memory_total_kb >= memory_cap_fraction:
        raise RuntimeError("measured CITE-seq peak RSS exceeded the configured cap")

    tables = {
        "dataset_inventory.tsv": inventory,
        "metrics_by_dataset.tsv": metrics,
        "method_summary.tsv": summary,
        "skipped_arms.tsv": skipped,
        "runtime_memory.tsv": runtime,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, sep="\t", index=False)
    report = _build_report(
        inventory, metrics, summary, skipped, max_peak_rss_kb, memory_total_kb
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    inputs = [dataset_manifest, Path(__file__).resolve(), *evaluation_manifests]
    outputs = [output_dir / filename for filename in tables]
    outputs.append(output_dir / "REPORT.md")
    manifest = {
        "schema_version": "crychic-citeseq-extended-report-v1",
        "status": "complete",
        "datasets": inventory["dataset"].tolist(),
        "dataset_count": len(inventory),
        "species_counts": inventory["species"].value_counts().sort_index().to_dict(),
        "comparison_contract": {
            "primary": "H-common/resource_fixed within species resource",
            "descriptive_only": [
                "H-common/independent",
                "native/independent",
                "cross-species method means",
            ],
        },
        "memory_audit": {
            "configured_cap_fraction": memory_cap_fraction,
            "host_memory_total_kb": memory_total_kb,
            "max_measured_peak_rss_kb": max_peak_rss_kb,
            "max_measured_fraction": max_peak_rss_kb / memory_total_kb,
            "within_cap": True,
        },
        "code": _git_state(workspace / "CRYCHIC"),
        "inputs": [_artifact(path, workspace) for path in inputs],
        "outputs": [_artifact(path, workspace) for path in outputs],
        "warnings": [
            "native resources differ across methods and are not directly comparable",
            "mouse CellPhoneDB is unavailable and excluded",
            "CBMC H-common has zero receptor-truth coverage and is not evaluable",
            "single-context CITE-seq cannot estimate differential communication",
            "ADT truth validates receptor specificity, not complete LR causality",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".."))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path(
            "../benchmark_work/literature_reproduction/citeseq/extended_datasets.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "../benchmark_work/literature_reproduction/reports/"
            "citeseq_extended_7datasets"
        ),
    )
    parser.add_argument("--memory-cap-fraction", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = summarize(
        workspace=args.workspace,
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
        memory_cap_fraction=args.memory_cap_fraction,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
