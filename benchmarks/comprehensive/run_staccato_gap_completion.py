"""Run frozen STACCato batch, calibration, interval, and power diagnostics."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA = "crychic-staccato-gap-completion-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _run(command: Sequence[str], label: str) -> tuple[str, float, str]:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    log = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"{label} failed: {log[-4000:]}")
    return label, time.perf_counter() - started, log


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.contract.read_text(encoding="utf-8"))["staccato"]
    for path in (args.rscript, args.r_worker, args.staccato_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.r_library.is_dir():
        raise FileNotFoundError(args.r_library)
    args.output.mkdir(parents=True)
    for name in ("workers", "logs", "timing"):
        (args.output / name).mkdir()
    replicates = int(config["simulation_replicates"])
    bootstrap = int(config["bootstrap_replicates"])
    settings: list[tuple[str, int, float, str]] = []
    for design in ("balanced", "moderate", "extreme"):
        settings.append((design, 60, 1.0, "condition_batch"))
        settings.append((design, 60, 0.0, "global_null"))
    for subjects in config["power_subject_counts"]:
        for effect in config["power_effect_multipliers"]:
            settings.append(("balanced", int(subjects), float(effect), "power_grid"))
    jobs: list[dict[str, Any]] = []
    for setting_index, (design, subjects, effect, campaign) in enumerate(settings):
        for replicate in range(1, replicates + 1):
            seed = args.first_seed + setting_index * 10_000 + replicate
            effect_label = str(effect).replace(".", "p")
            label = (
                f"{campaign}__{design}__n{subjects}__e{effect_label}"
                f"__rep{replicate:02d}__seed{seed}"
            )
            output = args.output / "workers" / f"{label}.tsv"
            timing = args.output / "timing" / f"{label}.tsv"
            command = [
                "/usr/bin/time",
                "-f",
                "%e\t%M",
                "-o",
                str(timing),
                "env",
                f"R_LIBS_USER={args.r_library}",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                str(args.rscript),
                str(args.r_worker),
                "--seed",
                str(seed),
                "--output",
                str(output),
                "--staccato-source",
                str(args.staccato_source),
                "--design",
                design,
                "--subjects",
                str(subjects),
                "--effect-multiplier",
                str(effect),
                "--bootstrap",
                str(bootstrap),
            ]
            jobs.append({"label": label, "command": command, "campaign": campaign})
    pd.DataFrame(
        [
            {
                "label": job["label"],
                "campaign": job["campaign"],
                "command": " ".join(job["command"]),
            }
            for job in jobs
        ]
    ).to_csv(args.output / "commands.tsv", sep="\t", index=False)
    started = time.perf_counter()
    runtimes: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run, job["command"], job["label"]): job for job in jobs
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            label, elapsed, log = future.result()
            runtimes[label] = elapsed
            (args.output / "logs" / f"{label}.log").write_text(log, encoding="utf-8")
            print(f"[{index}/{len(jobs)}] {label} {elapsed:.1f}s", flush=True)
    tables: list[pd.DataFrame] = []
    campaign_by_label = {job["label"]: job["campaign"] for job in jobs}
    for job in jobs:
        label = job["label"]
        table = pd.read_csv(args.output / "workers" / f"{label}.tsv", sep="\t")
        seconds, peak = (
            (args.output / "timing" / f"{label}.tsv").read_text().strip().split("\t")
        )
        table.insert(0, "campaign", campaign_by_label[label])
        table["worker_wall_seconds"] = runtimes[label]
        table["rscript_seconds"] = float(seconds)
        table["peak_rss_kib"] = int(peak)
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(args.output / "replicate_metrics.tsv", sep="\t", index=False)
    metric_columns = [
        column
        for column in metrics.columns
        if column
        not in {
            "campaign",
            "seed",
            "design",
            "subjects",
            "effect_multiplier",
            "bootstrap_replicates",
            "status",
        }
    ]
    aggregate = metrics.groupby(
        ["campaign", "design", "subjects", "effect_multiplier"], as_index=False
    ).agg(
        replicates=("seed", "size"),
        **{f"{column}_mean": (column, "mean") for column in metric_columns},
    )
    aggregate.to_csv(args.output / "aggregate_metrics.tsv", sep="\t", index=False)
    coverage = pd.DataFrame.from_records(
        [
            {
                "endpoint": "disease_and_batch_effect_error",
                "status": "complete",
                "reason_code": "",
            },
            {
                "endpoint": "null_type1_and_empirical_fdr",
                "status": "complete_diagnostic",
                "reason_code": "model_stage_residual_bootstrap",
            },
            {
                "endpoint": "interval_coverage_and_width",
                "status": "complete_diagnostic",
                "reason_code": "model_stage_residual_bootstrap",
            },
            {
                "endpoint": "power_by_effect_and_subject_count",
                "status": "complete_diagnostic",
                "reason_code": "model_stage_residual_bootstrap",
            },
            {
                "endpoint": "formal_full_pipeline_p_q",
                "status": "NE",
                "reason_code": "upstream_score_generation_not_resampled",
            },
            {
                "endpoint": "CRYCHIC_common_functional_OOF",
                "status": "NE",
                "reason_code": "unpaired_subject_context_design",
            },
        ]
    )
    coverage.to_csv(args.output / "endpoint_coverage.tsv", sep="\t", index=False)
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete_with_declared_NE",
            "protocol_status": "paper_design_public_low_rank_generator_reconstruction",
            "inference_scope": config["inference_scope"],
            "source_repository": git_metadata(args.repo_root),
            "contract": {
                "path": str(args.contract.resolve()),
                "sha256": sha256_file(args.contract),
            },
            "native_software": {
                "r_worker_sha256": sha256_file(args.r_worker),
                "staccato_source_sha256": sha256_file(args.staccato_source),
            },
            "execution": {
                "workers": args.workers,
                "jobs": len(jobs),
                "bootstrap_per_job": bootstrap,
                "campaign_wall_seconds": time.perf_counter() - started,
                "peak_worker_rss_kib": int(metrics["peak_rss_kib"].max()),
                "python": sys.version,
                "platform": platform.platform(),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)}
                for name in (
                    "commands.tsv",
                    "replicate_metrics.tsv",
                    "aggregate_metrics.tsv",
                    "endpoint_coverage.tsv",
                )
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_gap_completion_v1.json",
    )
    parser.add_argument(
        "--rscript", type=Path, default=Path("/home/agent1/miniforge3/bin/Rscript")
    )
    parser.add_argument(
        "--r-worker",
        type=Path,
        default=Path(__file__).with_name("staccato_gap_worker.R"),
    )
    parser.add_argument("--staccato-source", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 24))
    parser.add_argument("--first-seed", type=int, default=20_261_000)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
