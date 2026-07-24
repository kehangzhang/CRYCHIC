"""Run and provenance-bind the released scACCorDiON TCGA-PAAD survival protocol."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA = "crychic-scaccordion-tcga-paad-survival-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    source_commit = subprocess.run(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = "1850e4720a6acb1a7851831a6b6b7c83406bc8dc"
    if source_commit != expected:
        raise ValueError(f"unexpected scACCorDiON.su commit: {source_commit}")
    started = time.perf_counter()
    command = [
        str(args.rscript),
        str(args.r_worker),
        "--source-root",
        str(args.source_root.resolve()),
        "--differential",
        str(args.differential.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    (args.output / "execution.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    names = (
        "selected_lr_pairs.tsv",
        "cox_coefficients.tsv",
        "model_diagnostics.tsv",
        "proportional_hazards.tsv",
        "run_summary.tsv",
        "execution.log",
    )
    source_files = sorted((args.source_root / "data").glob("*.rda"))
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "protocol": (
                "released scACCorDiON.su custom top-10 LR selection with "
                "stage-adjusted Cox models"
            ),
            "source_repository": git_metadata(args.repo_root),
            "upstream": {
                "repository": "https://github.com/CostaLab/scACCorDiON.su",
                "commit": source_commit,
                "data": {path.name: sha256_file(path) for path in source_files},
                "differential_sha256": sha256_file(args.differential),
            },
            "execution": {
                "wall_seconds": time.perf_counter() - started,
                "R": platform.platform(),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)} for name in names
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--differential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, default=Path("/usr/bin/Rscript"))
    parser.add_argument(
        "--r-worker",
        type=Path,
        default=Path(__file__).with_name("scaccordion_survival_worker.R"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
