#!/usr/bin/env python3
"""Publish a path-safe, checksum-bound literature benchmark result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

JOINT_REPORT_DIR = Path("reports/literature_extension_20260717")
CITESEQ_REPORT_DIR = Path("reports/citeseq_extended_7datasets")
HER2_DIR = Path("cytokine/her2")
IPF_DIR = Path("results/ipf/full_cohort")

COPIED_TABLES = {
    "tables/citeseq_extension_summary.tsv": (
        JOINT_REPORT_DIR / "tables/citeseq_extension_summary.tsv"
    ),
    "tables/cytosig_extension_summary.tsv": (
        JOINT_REPORT_DIR / "tables/cytosig_extension_summary.tsv"
    ),
    "tables/ipf_crychic_by_study.tsv": (
        JOINT_REPORT_DIR / "tables/ipf_crychic_by_study.tsv"
    ),
    "tables/ipf_primary_metrics.tsv": (
        JOINT_REPORT_DIR / "tables/ipf_primary_metrics.tsv"
    ),
    "citeseq/dataset_inventory.tsv": CITESEQ_REPORT_DIR / "dataset_inventory.tsv",
    "citeseq/metrics_by_dataset.tsv": CITESEQ_REPORT_DIR / "metrics_by_dataset.tsv",
    "citeseq/method_summary.tsv": CITESEQ_REPORT_DIR / "method_summary.tsv",
    "citeseq/skipped_arms.tsv": CITESEQ_REPORT_DIR / "skipped_arms.tsv",
    "citeseq/runtime_memory.tsv": CITESEQ_REPORT_DIR / "runtime_memory.tsv",
    "cytosig_her2/ranking_metrics.tsv": HER2_DIR / "evaluation/ranking_metrics.tsv",
    "cytosig_her2/fisher_rank_curves.tsv": (
        HER2_DIR / "evaluation/fisher_rank_curves.tsv"
    ),
    "cytosig_her2/score_inventory.tsv": HER2_DIR / "evaluation/score_inventory.tsv",
    "ipf/cohort_patient_equal.tsv": IPF_DIR / "cohort_summary/cohort_patient_equal.tsv",
    "ipf/metrics_by_study.tsv": IPF_DIR / "cohort_summary/metrics_by_study.tsv",
    "ipf/patient_bootstrap.tsv": IPF_DIR / "cohort_summary/patient_bootstrap.tsv",
}

SOURCE_MANIFESTS = {
    "joint": JOINT_REPORT_DIR / "manifest.json",
    "citeseq": CITESEQ_REPORT_DIR / "manifest.json",
    "her2_preparation": HER2_DIR / "prepared/manifest.json",
    "her2_evaluation": HER2_DIR / "evaluation/manifest.json",
    "ipf": IPF_DIR / "manifest.json",
    "ipf_summary": IPF_DIR / "cohort_summary/manifest.json",
}

TOP_LINK_REWRITES = {
    "../../results/citeseq/cbmc/native/evaluation_independent/point_estimates.tsv": (
        "citeseq/metrics_by_dataset.tsv"
    ),
    "../../results/citeseq/cbmc/H-common/evaluation_resource_fixed/manifest.json": (
        "citeseq/manifest.json"
    ),
    (
        "../../results/citeseq/sln_111/H-common/evaluation_resource_fixed/"
        "point_estimates.tsv"
    ): ("citeseq/metrics_by_dataset.tsv"),
    (
        "../../results/citeseq/sln_208/H-common/evaluation_resource_fixed/"
        "point_estimates.tsv"
    ): ("citeseq/metrics_by_dataset.tsv"),
    "../../cytokine/her2/prepared/official_figure6a_reference.tsv": (
        "cytosig_her2/manifest.json"
    ),
    "../../cytokine/her2/REPORT.md": "cytosig_her2/README.md",
    "../../cytokine/her2/prepared/manifest.json": "cytosig_her2/manifest.json",
    "../../cytokine/her2/evaluation/ranking_metrics.tsv": (
        "cytosig_her2/ranking_metrics.tsv"
    ),
    "../../cytokine/her2/evaluation/fisher_rank_curves.tsv": (
        "cytosig_her2/fisher_rank_curves.tsv"
    ),
    "../../results/ipf/full_cohort/full_task_ledger.tsv": "ipf/manifest.json",
    "../../results/ipf/full_cohort/cohort_summary/manifest.json": ("ipf/manifest.json"),
    "../../results/ipf/full_cohort/cohort_summary/cohort_patient_equal.tsv": (
        "ipf/cohort_patient_equal.tsv"
    ),
    "../../results/ipf/full_cohort/cohort_summary/metrics_by_study.tsv": (
        "ipf/metrics_by_study.tsv"
    ),
    "../../results/ipf/full_cohort/cohort_summary/patient_bootstrap.tsv": (
        "ipf/patient_bootstrap.tsv"
    ),
    "../../ipf/parsed/gold_standard/ipf_gold_summary.json": "ipf/manifest.json",
}

FORBIDDEN_SUFFIXES = {
    ".h5ad",
    ".h5",
    ".log",
    ".parquet",
    ".rds",
    ".xlsx",
    ".xls",
}
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?!/)"
    r"(?:[A-Za-z0-9._~+-]+/)*[A-Za-z0-9._~+-]+"
)
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _count_tsv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return max(sum(1 for _ in reader) - 1, 0)


def _artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".tsv":
        record["rows"] = _count_tsv_rows(path)
    return record


def _source_record(source_root: Path, relative: Path) -> dict[str, Any]:
    path = source_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark artifact is missing: {relative}")
    return {
        "bytes": path.stat().st_size,
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
    }


def _copy_table(
    source_root: Path, output_root: Path, source: Path, output: str
) -> None:
    source_path = source_root / source
    if not source_path.is_file():
        raise FileNotFoundError(f"required benchmark table is missing: {source}")
    destination = output_root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)


def _contains_absolute_path(value: str) -> bool:
    stripped = value.strip().strip("\"'")
    return (
        stripped.startswith("/")
        or bool(POSIX_ABSOLUTE_PATH.search(value))
        or bool(WINDOWS_PATH.search(value))
    )


def _looks_command_column(column: str) -> bool:
    normalized = column.strip().lower()
    return (
        normalized in {"argv", "cmd", "command", "shell_command"}
        or normalized.startswith(("argv_", "cmd_", "command_"))
        or normalized.endswith(("_argv", "_cmd", "_command"))
    )


def sanitize_tsv(source: Path, destination: Path) -> list[str]:
    """Remove command columns and columns containing absolute machine paths."""

    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {source}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    dropped = {
        column
        for column in fieldnames
        if _looks_command_column(column)
        or any(_contains_absolute_path(row.get(column, "")) for row in rows)
    }
    kept = [column for column in fieldnames if column not in dropped]
    if not kept:
        raise ValueError(f"sanitization removed every column from {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=kept,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return sorted(dropped)


def _rewrite_top_report(report: str) -> str:
    for old, new in TOP_LINK_REWRITES.items():
        report = report.replace(f"]({old})", f"]({new})")
    report = report.replace("`/usr/bin/time -v`", "`GNU time -v`")
    title, remainder = report.split("\n", maxsplit=1)
    notice = (
        "> GitHub 发布副本: 仅包含汇总表、可审计的小型结果表和清理后的发布证明。"
        "原始 H5AD、Parquet、RDS、XLSX、日志、任务命令及机器绝对路径均未纳入。"
    )
    return f"{title}\n\n{notice}\n{remainder}"


def _append_artifact_links(report: str, lines: list[str]) -> str:
    return report.rstrip() + "\n\n## Published artifacts\n\n" + "\n".join(lines) + "\n"


def _build_readmes(source_root: Path, output_root: Path) -> None:
    joint = (source_root / JOINT_REPORT_DIR / "REPORT.md").read_text(encoding="utf-8")
    (output_root / "README.md").write_text(_rewrite_top_report(joint), encoding="utf-8")

    citeseq = (source_root / CITESEQ_REPORT_DIR / "REPORT.md").read_text(
        encoding="utf-8"
    )
    citeseq = _append_artifact_links(
        citeseq,
        [
            "- [Dataset inventory](dataset_inventory.tsv)",
            "- [All dataset-level metrics](metrics_by_dataset.tsv)",
            "- [Method summary](method_summary.tsv)",
            "- [Skipped and non-evaluable arms](skipped_arms.tsv)",
            "- [Runtime and memory observations](runtime_memory.tsv)",
            "- [Publication manifest](manifest.json)",
        ],
    )
    (output_root / "citeseq/README.md").write_text(citeseq, encoding="utf-8")

    her2 = (source_root / HER2_DIR / "REPORT.md").read_text(encoding="utf-8")
    her2 = her2.split("## Checksum linkage", maxsplit=1)[0]
    her2 = _append_artifact_links(
        her2,
        [
            "- [Ranking metrics](ranking_metrics.tsv)",
            "- [Fisher rank curves](fisher_rank_curves.tsv)",
            "- [Score inventory](score_inventory.tsv)",
            "- [Sanitized provenance manifest](manifest.json)",
            "",
            "Large inputs and score matrices are checksum-referenced in the manifest "
            "but intentionally excluded from this GitHub bundle.",
        ],
    )
    (output_root / "cytosig_her2/README.md").write_text(her2, encoding="utf-8")

    ipf = (source_root / IPF_DIR / "REPORT.md").read_text(encoding="utf-8")
    ipf = ipf.replace("`/usr/bin/time -v`", "`GNU time -v`")
    ipf = ipf.split("## 可复现产物", maxsplit=1)[0]
    ipf = _append_artifact_links(
        ipf,
        [
            "- [患者等权汇总](cohort_patient_equal.tsv)",
            "- [Study 分层汇总](metrics_by_study.tsv)",
            "- [患者 bootstrap 汇总](patient_bootstrap.tsv)",
            "- [清理后的样本级结果](sample_metrics.tsv)",
            "- [发布证明](manifest.json)",
        ],
    )
    (output_root / "ipf/README.md").write_text(ipf, encoding="utf-8")


def _records_for(output_root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    return {
        relative: _artifact_record(output_root / relative)
        for relative in sorted(relative_paths)
    }


def _write_track_manifests(
    source_root: Path,
    output_root: Path,
    source_manifests: dict[str, dict[str, Any]],
    dropped_ipf_columns: list[str],
) -> None:
    cite_source = source_manifests["citeseq"]
    cite_outputs = [
        "citeseq/README.md",
        "citeseq/dataset_inventory.tsv",
        "citeseq/metrics_by_dataset.tsv",
        "citeseq/method_summary.tsv",
        "citeseq/skipped_arms.tsv",
        "citeseq/runtime_memory.tsv",
    ]
    _write_json(
        output_root / "citeseq/manifest.json",
        {
            "schema_version": "crychic-citeseq-publication-v1",
            "status": "complete",
            "repository_state_at_run": cite_source.get("code", {}),
            "comparison_contract": cite_source.get("comparison_contract", {}),
            "dataset_count": cite_source.get("dataset_count"),
            "datasets": cite_source.get("datasets", []),
            "source_manifest": _source_record(source_root, SOURCE_MANIFESTS["citeseq"]),
            "outputs": _records_for(output_root, cite_outputs),
        },
    )

    preparation = source_manifests["her2_preparation"]
    evaluation = source_manifests["her2_evaluation"]
    her2_outputs = [
        "cytosig_her2/README.md",
        "cytosig_her2/fisher_rank_curves.tsv",
        "cytosig_her2/ranking_metrics.tsv",
        "cytosig_her2/score_inventory.tsv",
    ]
    _write_json(
        output_root / "cytosig_her2/manifest.json",
        {
            "schema_version": "crychic-cytosig-her2-publication-v1",
            "status": "complete",
            "dataset_id": preparation.get("dataset_id", "Wu_GSE176078_HER2"),
            "truth_status": preparation.get("truth_status"),
            "limitations": evaluation.get("limitations", []),
            "source_manifests": {
                name: _source_record(source_root, SOURCE_MANIFESTS[name])
                for name in ("her2_preparation", "her2_evaluation")
            },
            "excluded_source_artifact_sha256": {
                "prepared_h5ad": (
                    "f13153c1ddebba798635f841bc51a080abedcceb9cd53cb0269717024e30df69"
                ),
                "official_source_xlsx": (
                    "4d2848bc91a7082a36bd90bf4d5c9d0136eee23d748af055c0717af62f231b5a"
                ),
                "crychic_scores_parquet": (
                    "ad1e9fba506ed5bb1e39f97f4e90586fd722839d7256951d37655d6e5c4c2448"
                ),
                "liana_scores_parquet": (
                    "a83f60340e8205c6d0d552a5b7087ed3fa09605266b246bdb54b82fef92d68fa"
                ),
            },
            "outputs": _records_for(output_root, her2_outputs),
        },
    )

    ipf_source = source_manifests["ipf"]
    ipf_outputs = [
        "ipf/README.md",
        "ipf/cohort_patient_equal.tsv",
        "ipf/metrics_by_study.tsv",
        "ipf/patient_bootstrap.tsv",
        "ipf/sample_metrics.tsv",
    ]
    frozen_sha = {
        name: details.get("sha256")
        for name, details in ipf_source.get("frozen_inputs", {}).items()
    }
    artifacts = ipf_source.get("artifacts", {})
    _write_json(
        output_root / "ipf/manifest.json",
        {
            "schema_version": "crychic-ipf-publication-v1",
            "status": "complete",
            "source_record": ipf_source.get("source_record"),
            "cohort": ipf_source.get("cohort", {}),
            "execution": ipf_source.get("execution", {}),
            "evaluation_attestation": ipf_source.get("evaluation_attestation", {}),
            "repository_state_at_run": ipf_source.get("repository_state", {}),
            "frozen_input_sha256": frozen_sha,
            "source_attestations": {
                "full_task_ledger_sha256": artifacts.get("full_task_ledger.tsv"),
                "evaluation_manifest_aggregate_sha256": ipf_source.get(
                    "evaluation_attestation", {}
                ).get("evaluation_manifest_aggregate_sha256"),
            },
            "sanitization": {
                "sample_metrics_dropped_columns": dropped_ipf_columns,
            },
            "limitations": ipf_source.get("limitations", []),
            "source_manifests": {
                name: _source_record(source_root, SOURCE_MANIFESTS[name])
                for name in ("ipf", "ipf_summary")
            },
            "outputs": _records_for(output_root, ipf_outputs),
        },
    )


def validate_bundle(root: Path) -> None:
    """Reject disallowed files, host paths, and unresolved local Markdown links."""

    if not root.is_dir():
        raise ValueError(f"publication bundle does not exist: {root}")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden artifact type in publication bundle: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"publication artifact is not UTF-8 text: {path}"
            ) from error
        if POSIX_ABSOLUTE_PATH.search(text) or WINDOWS_PATH.search(text):
            raise ValueError(
                f"absolute machine path found in publication bundle: {path}"
            )
        if path.suffix.lower() != ".md":
            continue
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            linked = (path.parent / target).resolve()
            if root.resolve() not in (linked, *linked.parents):
                raise ValueError(
                    f"Markdown link escapes publication bundle: {path}: {target}"
                )
            if not linked.exists():
                raise ValueError(f"unresolved Markdown link: {path}: {target}")


def publish_bundle(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Build and validate the repository-safe publication directory."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_manifests = {
        name: _read_json(source_root / relative)
        for name, relative in SOURCE_MANIFESTS.items()
    }
    for source in COPIED_TABLES.values():
        if not (source_root / source).is_file():
            raise FileNotFoundError(f"required benchmark table is missing: {source}")

    temporary = output_root.with_name(f".{output_root.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for output, source in COPIED_TABLES.items():
            _copy_table(source_root, temporary, source, output)
        dropped_ipf_columns = sanitize_tsv(
            source_root / IPF_DIR / "cohort_summary/sample_metrics.tsv",
            temporary / "ipf/sample_metrics.tsv",
        )
        _build_readmes(source_root, temporary)
        _write_track_manifests(
            source_root,
            temporary,
            source_manifests,
            dropped_ipf_columns,
        )

        source_inputs = {
            relative.as_posix(): _source_record(source_root, relative)
            for relative in sorted(
                {
                    *COPIED_TABLES.values(),
                    *(path for path in SOURCE_MANIFESTS.values()),
                    JOINT_REPORT_DIR / "REPORT.md",
                    CITESEQ_REPORT_DIR / "REPORT.md",
                    HER2_DIR / "REPORT.md",
                    IPF_DIR / "REPORT.md",
                    IPF_DIR / "cohort_summary/sample_metrics.tsv",
                },
                key=lambda item: item.as_posix(),
            )
        }
        generated = [
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file()
        ]
        joint = source_manifests["joint"]
        cite_code = source_manifests["citeseq"].get("code", {})
        manifest = {
            "schema_version": "crychic-literature-extension-publication-v1",
            "status": "complete",
            "generated_on": joint.get("generated_on", "2026-07-17"),
            "headline": joint.get("headline", {}),
            "repository_state_at_run": cite_code,
            "safety": {
                "absolute_machine_paths_in_bundle": False,
                "all_markdown_links_resolve": True,
                "excluded_artifact_types": sorted(FORBIDDEN_SUFFIXES),
                "ipf_sample_metrics_dropped_columns": dropped_ipf_columns,
                "raw_logs_and_task_commands_excluded": True,
            },
            "inputs": source_inputs,
            "outputs": _records_for(temporary, generated),
        }
        _write_json(temporary / "manifest.json", manifest)
        validate_bundle(temporary)

        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to benchmark_work/literature_reproduction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Repository destination for the publication bundle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = publish_bundle(args.source_root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": manifest["status"],
                "files": len(manifest["outputs"]) + 1,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
