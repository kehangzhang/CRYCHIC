"""Publish compact, checksum-bound summaries from component-swap runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)

SCHEMA_VERSION = "crychic-component-swap-summary-v1"
TABLES = (
    "spatial_des_summary.tsv",
    "arm_availability.tsv",
    "arm_diagnostics.tsv",
    "pairwise_diagnostics.tsv",
    "threshold_path_summary.tsv",
    "pair_rank_comparison.tsv",
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return cast(dict[str, Any], value)


def _bound_table(run: Path, manifest: Mapping[str, object], name: str) -> Path:
    outputs = manifest.get("outputs")
    record = outputs.get(name) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"component-swap manifest does not bind {name}")
    path = run / name
    if (
        not path.is_file()
        or record.get("filename") != name
        or record.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"component-swap output binding is invalid for {name}")
    return path


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def run(
    runs: Mapping[str, Path], output_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    if set(runs) != {"kuppe", "ms"}:
        raise ValueError("runs must contain exactly kuppe and ms")
    output = prepare_output(output_dir, overwrite=overwrite)
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    coverage_tables: list[pd.DataFrame] = []
    source_records: dict[str, object] = {}
    for dataset, run_dir in sorted(runs.items()):
        run_path = run_dir.resolve()
        manifest_path = run_path / "manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") != "complete"
            or manifest.get("dataset") != dataset
            or manifest.get("algorithm_modified") is not False
            or manifest.get("arm_a_parity") != {
                "exact": True,
                "rows": 132 if dataset == "kuppe" else 90,
            }
        ):
            raise ValueError(f"invalid completed component-swap run: {dataset}")
        bound_outputs: dict[str, object] = {}
        for name in TABLES:
            path = _bound_table(run_path, manifest, name)
            table = pd.read_csv(path, sep="\t")
            table.insert(0, "benchmark_dataset", dataset)
            collected[name].append(table)
            bound_outputs[name] = sha256_file(path)
        coverage_path = _bound_table(
            run_path, manifest, "spatial_des_coverage.tsv"
        )
        coverage = pd.read_csv(coverage_path, sep="\t")
        coverage.insert(0, "benchmark_dataset", dataset)
        coverage_tables.append(coverage)
        source_records[dataset] = {
            "manifest_sha256": sha256_file(manifest_path),
            "arm_a_parity": manifest["arm_a_parity"],
            "implementation": manifest["implementation"],
            "compact_output_sha256": bound_outputs,
            "coverage_sha256": sha256_file(coverage_path),
        }

    tables = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in collected.items()
    }
    des = tables["spatial_des_summary.tsv"]
    des["complete_8_of_8"] = des["count"].eq(8)
    a_median = (
        des.loc[des["method"].eq("A"), ["benchmark_dataset", "median"]]
        .rename(columns={"median": "arm_a_median"})
        .set_index("benchmark_dataset")["arm_a_median"]
    )
    des["delta_median_vs_A"] = des["median"] - des["benchmark_dataset"].map(
        a_median
    )
    des["complete_rank"] = np.nan
    complete = des["complete_8_of_8"]
    des.loc[complete, "complete_rank"] = (
        des.loc[complete]
        .groupby("benchmark_dataset", observed=True, sort=False)["median"]
        .rank(method="min", ascending=False)
    )

    coverage = pd.concat(coverage_tables, ignore_index=True)
    coverage_summary = (
        coverage.groupby(
            ["benchmark_dataset", "method"], observed=True, sort=True
        )
        .agg(
            strata=("status", "size"),
            observed_strata=("status", lambda value: int(value.eq("observed").sum())),
            minimum_expected_pair_coverage=(
                "expected_pair_coverage_fraction",
                "min",
            ),
            minimum_ranked_pairs=("rank_pairs_eligible", "min"),
            maximum_ranked_pairs=("rank_pairs_eligible", "max"),
        )
        .reset_index()
    )
    coverage_summary["complete_8_of_8"] = coverage_summary[
        "observed_strata"
    ].eq(8)
    tables["coverage_summary.tsv"] = coverage_summary
    tables["threshold_path_summary.tsv"]["post_hoc_diagnostic_only"] = True

    outputs: dict[str, object] = {}
    for name, table in tables.items():
        path = output / name
        table.to_csv(path, sep="\t", index=False, lineterminator="\n")
        outputs[name] = _output_record(path, table)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm_modified": False,
        "source_runs": source_records,
        "implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "code": git_metadata(Path(__file__).resolve().parents[2]),
        },
        "outputs": outputs,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _parse_run(value: str) -> tuple[str, Path]:
    dataset, separator, path = value.partition("=")
    if not separator or dataset not in {"kuppe", "ms"} or not path:
        raise argparse.ArgumentTypeError("--run must be kuppe=PATH or ms=PATH")
    return dataset, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    runs = dict(args.run)
    if len(runs) != len(args.run):
        parser.error("duplicate --run dataset")
    result = run(runs, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"status": result["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
