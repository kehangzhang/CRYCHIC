"""Build a deterministic compact archive of benchmark notebooks and source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BUNDLE_DIR = "CRYCHIC_benchmark_bundle_20260724"
FIXED_ZIP_TIME = (2026, 7, 24, 0, 0, 0)

NOTEBOOKS = {
    "tutorials/literature_benchmark_results.ipynb": (
        "notebooks/single_group/literature_benchmark_results.ipynb"
    ),
    "tutorials/suggest_v5_under100k_benchmark_results.ipynb": (
        "notebooks/multigroup/suggest_v5_under100k_benchmark_results.ipynb"
    ),
}

SINGLE_FIGURES = (
    "citeseq_fixed_resource_metrics.png",
    "cytosig_controlled_metrics.png",
    "cytosig_top250_odds_ratio.png",
    "ipf_patient_equal_methods.png",
    "ipf_study_and_bootstrap.png",
    "latest_single_sample_consistency.png",
    "robustness_macro_auc.png",
    "robustness_recovery_curves.png",
    "suggest_v5_gap_completion.png",
)

MULTIGROUP_FIGURES = tuple(
    f"figure_{index:02d}_{name}.png"
    for index, name in enumerate(
        (
            "tensor_cell2cell",
            "staccato",
            "dcst",
            "patient_graphs",
            "rcc_annotation",
            "native_three_group",
            "score_engine_crossover",
        ),
        start=1,
    )
)

REPORTS = {
    "docs/benchmarks/compact_single_multigroup_benchmark_summary_20260724.md": (
        "README_BENCHMARK_SUMMARY.md"
    ),
    (
        "docs/benchmarks/"
        "suggest_v5_gap_completion_and_single_sample_consistency_20260724.md"
    ): (
        "reports/suggest_v5_gap_completion_and_single_sample_consistency.md"
    ),
    "benchmarks/results/SUGGEST_V6_BOUNDED_EVIDENCE_REPORT_20260724.md": (
        "reports/SUGGEST_V6_BOUNDED_EVIDENCE_REPORT.md"
    ),
    "benchmarks/results/suggest_v6_bounded_evidence_summary_20260724.json": (
        "reports/suggest_v6_bounded_evidence_summary.json"
    ),
    "benchmarks/results/suggest_v5_final_summary_20260723.json": (
        "reports/suggest_v5_final_summary.json"
    ),
}


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tracked_source_files(repo_root: Path) -> list[Path]:
    tracked = _git(repo_root, "ls-files", "-z").split("\0")
    selected: list[Path] = []
    benchmark_suffixes = {".py", ".R"}
    config_suffixes = {".json", ".yaml", ".yml"}
    for relative_text in tracked:
        if not relative_text:
            continue
        relative = Path(relative_text)
        if relative.parts[:2] == ("src", "crychic"):
            selected.append(relative)
        elif relative.parts[:1] == ("benchmarks",) and relative.suffix in (
            benchmark_suffixes
        ):
            selected.append(relative)
        elif relative.parts[:2] == ("benchmarks", "configs") and relative.suffix in (
            config_suffixes
        ):
            selected.append(relative)
    return sorted(set(selected), key=lambda path: path.as_posix())


def _validate_notebook(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in payload["cells"] if cell.get("cell_type") == "code"
    ]
    if not code_cells or any(
        cell.get("execution_count") is None for cell in code_cells
    ):
        raise ValueError(f"notebook is not fully executed: {path}")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise ValueError(f"notebook contains error outputs: {path}")
    images = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    return {
        "cells": len(payload["cells"]),
        "code_cells": len(code_cells),
        "embedded_png_outputs": images,
    }


def _zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(f"{BUNDLE_DIR}/{name}", FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _add_file(
    records: list[dict[str, Any]],
    payloads: dict[str, bytes],
    source: Path,
    destination: str,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    data = source.read_bytes()
    if destination in payloads:
        raise ValueError(f"duplicate archive destination: {destination}")
    payloads[destination] = data
    records.append(
        {"path": destination, "size_bytes": len(data), "sha256": _sha256(data)}
    )


def _find_benchmark_work(repo_root: Path) -> Path:
    for ancestor in (repo_root, *repo_root.parents):
        candidate = ancestor / "benchmark_work"
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("cannot locate benchmark_work from repository ancestors")


def build_bundle(
    *, repo_root: Path, benchmark_work: Path, output: Path, require_clean: bool
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    benchmark_work = benchmark_work.resolve()
    if require_clean and _git(repo_root, "status", "--porcelain=v1"):
        raise RuntimeError("refusing to package a dirty Git worktree")

    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    notebook_metadata: dict[str, dict[str, int]] = {}

    for source_text, destination in NOTEBOOKS.items():
        source = repo_root / source_text
        notebook_metadata[destination] = _validate_notebook(source)
        _add_file(records, payloads, source, destination)

    gap_root = benchmark_work / "suggest_v5_gap_completion_20260724"
    multi_root = benchmark_work / "suggest_v5_under100k_20260724"
    for name in SINGLE_FIGURES:
        _add_file(
            records,
            payloads,
            gap_root / "notebook_figures" / name,
            f"figures/single_group/{name}",
        )
    for name in MULTIGROUP_FIGURES:
        _add_file(
            records,
            payloads,
            multi_root / "publication_figures" / name,
            f"figures/multigroup/{name}",
        )

    for source_text, destination in REPORTS.items():
        _add_file(records, payloads, repo_root / source_text, destination)

    compact_results = {
        gap_root / "single_sample_consistency/track_summary.tsv": (
            "results_compact/single_sample_consistency.tsv"
        ),
        gap_root / "single_sample_consistency/score_head_coverage.tsv": (
            "results_compact/single_sample_score_head_coverage.tsv"
        ),
        gap_root / "scaccordion_crosscohort_v2/friedman_average_ranks.tsv": (
            "results_compact/scaccordion_friedman_average_ranks.tsv"
        ),
        gap_root / "dcst_binary/aggregate_metrics.tsv": (
            "results_compact/dcst_fixed_binary_aggregate.tsv"
        ),
    }
    for source, destination in compact_results.items():
        _add_file(records, payloads, source, destination)

    for relative in _tracked_source_files(repo_root):
        _add_file(
            records,
            payloads,
            repo_root / relative,
            f"source/{relative.as_posix()}",
        )
    for relative_text in ("README.md", "pyproject.toml", "uv.lock"):
        _add_file(
            records,
            payloads,
            repo_root / relative_text,
            f"source/{relative_text}",
        )

    records.sort(key=lambda row: str(row["path"]))
    metadata = {
        "schema_version": "crychic-compact-benchmark-bundle-v1",
        "generated_on": "2026-07-24",
        "git": {
            "branch": _git(repo_root, "branch", "--show-current"),
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "dirty": bool(_git(repo_root, "status", "--porcelain=v1")),
        },
        "notebooks": notebook_metadata,
        "standalone_png_figures": len(SINGLE_FIGURES) + len(MULTIGROUP_FIGURES),
        "payload_files_excluding_indexes": len(records),
        "uncompressed_payload_bytes_excluding_indexes": sum(
            int(row["size_bytes"]) for row in records
        ),
        "exclusions": [
            "raw and prepared datasets",
            "full benchmark result directories",
            "conda/mamba environments",
            "PDF and SVG figure duplicates",
            "Python bytecode and cache directories",
        ],
    }
    metadata_data = (
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    payloads["BUNDLE_METADATA.json"] = metadata_data
    records.append(
        {
            "path": "BUNDLE_METADATA.json",
            "size_bytes": len(metadata_data),
            "sha256": _sha256(metadata_data),
        }
    )
    index_lines = ["sha256\tsize_bytes\tpath"] + [
        f"{row['sha256']}\t{row['size_bytes']}\t{row['path']}"
        for row in sorted(records, key=lambda row: str(row["path"]))
    ]
    index_data = ("\n".join(index_lines) + "\n").encode("utf-8")
    payloads["BUNDLE_FILE_INDEX.tsv"] = index_data

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(output, mode="x", allowZip64=True) as archive:
        for destination in sorted(payloads):
            _zip_write(archive, destination, payloads[destination])

    archive_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{archive_sha256}  {output.name}\n", encoding="ascii"
    )
    return {
        **metadata,
        "archive": str(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": archive_sha256,
        "checksum_file": str(checksum_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    benchmark_work = (
        args.benchmark_work.resolve()
        if args.benchmark_work is not None
        else _find_benchmark_work(repo_root)
    )
    summary = build_bundle(
        repo_root=repo_root,
        benchmark_work=benchmark_work,
        output=args.output.resolve(),
        require_clean=not args.allow_dirty,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
