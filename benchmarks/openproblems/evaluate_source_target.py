"""Evaluate source-target predictions with Open Problems v1.0.0 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.openproblems.common import (
    OFFICIAL_RESULTS_URL,
    OFFICIAL_V1_RANDOM_AUPRC,
    OFFICIAL_V1_RANDOM_ODDS_RATIO,
    load_truth,
    official_metrics,
    precision_recall_points,
    sha256_file,
    validate_predictions,
    write_json,
)


def _prediction_paths(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    paths = sorted(
        {
            path.resolve()
            for root in roots
            for path in (
                (root,) if root.is_file() else tuple(root.rglob("*.parquet"))
            )
            if path.name != "crychic_availability_lr.parquet"
        }
    )
    if not paths:
        raise ValueError("no prediction parquet files were found")
    return tuple(paths)


def _local_metrics(
    truth: pd.DataFrame,
    labels: tuple[str, ...],
    paths: tuple[Path, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    metrics: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    method_ids: set[str] = set()
    for path in paths:
        table = pd.read_parquet(path)
        validate_predictions(table, labels=labels)
        method_id_values = table["method_id"].astype(str).unique()
        if len(method_id_values) != 1:
            raise ValueError(f"{path} contains more than one method ID")
        method_id = str(method_id_values[0])
        if method_id in method_ids:
            raise ValueError(f"duplicate local method ID: {method_id}")
        method_ids.add(method_id)
        values = official_metrics(truth, table)
        metadata = table.iloc[0]
        metrics.append(
            {
                "result_source": "local_current_implementation",
                "method_id": method_id,
                "method_name": str(metadata["method_name"]),
                "method_scope": str(metadata["method_scope"]),
                "resource_id": str(metadata["resource_id"]),
                "aggregation": str(metadata["aggregation"]),
                **values,
                "prediction_file": path.name,
                "prediction_sha256": sha256_file(path),
            }
        )
        curve = precision_recall_points(truth, table)
        curve.insert(0, "method_name", str(metadata["method_name"]))
        curve.insert(0, "method_id", method_id)
        curves.append(curve)
        hashes[str(path)] = sha256_file(path)
    return (
        pd.DataFrame.from_records(metrics),
        pd.concat(curves, ignore_index=True),
        hashes,
    )


def _official_reference(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("official results payload must be a list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if item.get("dataset_id") != "mouse_brain_atlas":
            continue
        raw = item["metric_values"]
        scaled = item["scaled_scores"]
        rows.append(
            {
                "result_source": "official_v1_reference_2023",
                "method_id": str(item["method_id"]),
                "method_name": str(item["method_id"]),
                "method_scope": "official_openproblems_v1_R_LIANA",
                "resource_id": "official_v1_embedded_resource",
                "aggregation": (
                    str(item["method_id"]).rsplit("_", maxsplit=1)[-1]
                    if str(item["method_id"]).rsplit("_", maxsplit=1)[-1]
                    in {"max", "sum"}
                    else "not_applicable"
                ),
                "openproblems_score": float(item["mean_score"]),
                "odds_ratio": float(scaled["odds_ratio"]),
                "precision_recall_auc": float(scaled["auprc"]),
                "odds_ratio_raw": float(raw["odds_ratio"]),
                "precision_recall_auc_raw": float(raw["auprc"]),
                "auroc_supplementary": np.nan,
                "average_precision_supplementary": np.nan,
                "random_odds_ratio_raw": OFFICIAL_V1_RANDOM_ODDS_RATIO,
                "random_precision_recall_auc_raw": OFFICIAL_V1_RANDOM_AUPRC,
                "prediction_file": pd.NA,
                "prediction_sha256": pd.NA,
                "official_commit_sha": str(item["commit_sha"]),
                "official_code_version": str(item["code_version"]),
            }
        )
    result = pd.DataFrame.from_records(rows)
    random = result.loc[result["method_id"].eq("random_events")]
    if len(random) != 1 or not np.allclose(
        random[["precision_recall_auc_raw", "odds_ratio_raw"]].to_numpy(float),
        [[OFFICIAL_V1_RANDOM_AUPRC, OFFICIAL_V1_RANDOM_ODDS_RATIO]],
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("official Random Events baseline differs from v1 constants")
    return result


def _json_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    safe = table.astype("object").where(pd.notna(table), None)
    records = safe.to_dict(orient="records")
    for record in records:
        for key, value in tuple(record.items()):
            if isinstance(value, float) and not np.isfinite(value):
                record[key] = None
    return records


def run(
    input_h5ad: Path,
    prediction_roots: tuple[Path, ...],
    official_results_json: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"evaluation already exists: {manifest_path}")
    truth, labels = load_truth(input_h5ad)
    paths = _prediction_paths(prediction_roots)
    local, curves, hashes = _local_metrics(truth, labels, paths)
    reference = _official_reference(official_results_json)
    combined = pd.concat([local, reference], ignore_index=True, sort=False)
    local_path = output_dir / "local_metrics.tsv"
    reference_path = output_dir / "official_v1_reference.tsv"
    combined_path = output_dir / "benchmark_metrics.tsv"
    curves_path = output_dir / "precision_recall_curves.parquet"
    local.to_csv(local_path, sep="\t", index=False, na_rep="")
    reference.to_csv(reference_path, sep="\t", index=False, na_rep="")
    combined.to_csv(combined_path, sep="\t", index=False, na_rep="")
    curves.to_parquet(curves_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": "crychic-openproblems-source-target-evaluation-v1",
        "status": "complete",
        "task_id": "cell_cell_communication_source_target",
        "task_version": "v1.0.0",
        "dataset": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "truth_rows": len(truth),
            "truth_positive": int(truth["response"].sum()),
            "labels": len(labels),
        },
        "metrics": {
            "primary": ["precision_recall_auc", "odds_ratio"],
            "composite": "openproblems_score",
            "raw_columns": [
                "precision_recall_auc_raw",
                "odds_ratio_raw",
            ],
            "scaling": (
                "(raw - official_random_v1) / (1 - official_random_v1); "
                "unclipped; mean of the two scaled primary metrics"
            ),
            "official_random_v1": {
                "precision_recall_auc_raw": OFFICIAL_V1_RANDOM_AUPRC,
                "odds_ratio_raw": OFFICIAL_V1_RANDOM_ODDS_RATIO,
            },
            "supplementary": [
                "auroc_supplementary",
                "average_precision_supplementary",
            ],
        },
        "official_reference": {
            "url": OFFICIAL_RESULTS_URL,
            "filename": official_results_json.name,
            "sha256": sha256_file(official_results_json),
            "rows": len(reference),
        },
        "local_predictions": hashes,
        "outputs": {
            path.name: sha256_file(path)
            for path in (local_path, reference_path, combined_path, curves_path)
        },
        "local_results": _json_records(local),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("official_results_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("prediction_root", nargs="+", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        tuple(args.prediction_root),
        args.official_results_json,
        args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
