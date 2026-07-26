"""Prepare subject-level scSeqCommDiff fixed-K and continuous DES rankings."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    sha256_file,
    write_json,
)
from benchmarks.literature.event_level_des import (
    EVENT_BUDGETS,
    mechanism_annotations,
    pair_rankings_from_events,
    prepare_scseqcommdiff_event_ledger,
    select_top_k_events_by_scope,
)
from benchmarks.literature.run_event_level_des_real import (
    SCSEQ_SCHEMA_VERSIONS,
    _bound_output,
    _export_scseq_ledger,
    _load_resource,
)

SCHEMA_VERSION = "crychic-scseqcommdiff-subject-event-tracks-v1"
TOP_K_SCOPE = "global_across_both_effect_directions"


def _read_json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _single_text(table: pd.DataFrame, column: str) -> str:
    if column not in table:
        raise ValueError(f"ranking axes lack {column!r}")
    values = table[column].dropna().astype(str).unique()
    if len(values) != 1 or not values[0] or values[0] != values[0].strip():
        raise ValueError(f"ranking axes require one canonical {column}")
    return str(values[0])


def build_subject_event_track_rankings(
    ledger: pd.DataFrame,
    pair_axes: pd.DataFrame,
    *,
    dataset_id: str,
    resource_id: str,
    event_budgets: Sequence[int] = EVENT_BUDGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact global fixed-K tracks and one continuous-strength track."""

    if tuple(event_budgets) != tuple(sorted(set(event_budgets))):
        raise ValueError("event budgets must be unique and ascending")
    eligible = ledger["top_k_eligible"].astype(bool)
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for budget in event_budgets:
        selected = select_top_k_events_by_scope(
            ledger,
            budget=budget,
            evidence_column="event_evidence",
            eligible=eligible,
            scope=TOP_K_SCOPE,
        )
        ranking = pair_rankings_from_events(
            ledger,
            pair_axes,
            selected=selected,
            weight_column=None,
            metadata={
                "dataset": dataset_id,
                "method": "scseqcommdiff",
                "method_version": "2.0.0",
                "resource": resource_id,
                "ranking_semantics": (
                    f"global_top_{budget}_negative_log10_native_p"
                ),
            },
        )
        ranking.insert(0, "des_variant", "top_k_count_des")
        ranking.insert(1, "event_budget", budget)
        parts.append(ranking)
        diagnostics.append(
            {
                "des_variant": "top_k_count_des",
                "event_budget": budget,
                "selected_events": int(selected.sum()),
                "rank_rows": len(ranking),
            }
        )

    continuous_selected = eligible & ledger["abs_effect"].gt(0.0)
    continuous = pair_rankings_from_events(
        ledger,
        pair_axes,
        selected=continuous_selected,
        weight_column="continuous_weight",
        metadata={
            "dataset": dataset_id,
            "method": "scseqcommdiff",
            "method_version": "2.0.0",
            "resource": resource_id,
            "ranking_semantics": "sum_abs_native_intercellular_score_difference",
        },
    )
    continuous.insert(0, "des_variant", "continuous_weighted_des")
    continuous.insert(1, "event_budget", pd.NA)
    parts.append(continuous)
    diagnostics.append(
        {
            "des_variant": "continuous_weighted_des",
            "event_budget": pd.NA,
            "selected_events": int(continuous_selected.sum()),
            "rank_rows": len(continuous),
        }
    )
    return (
        pd.concat(parts, ignore_index=True, sort=False),
        pd.DataFrame.from_records(diagnostics),
    )


def run(
    source_manifest_path: Path,
    resource_manifest_path: Path,
    rscript: Path,
    output_dir: Path,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Validate one subject run, export its events, and persist track rankings."""

    for path in (source_manifest_path, resource_manifest_path, rscript):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    code = git_metadata(Path(__file__).resolve().parents[2])
    if code["dirty"] and not allow_dirty:
        raise RuntimeError("subject event-track producer refuses a dirty worktree")

    source = _read_json_object(source_manifest_path)
    preflight = source.get("preflight")
    source_input = preflight.get("input") if isinstance(preflight, Mapping) else None
    source_resource = (
        preflight.get("resource") if isinstance(preflight, Mapping) else None
    )
    if (
        source.get("schema_version") not in SCSEQ_SCHEMA_VERSIONS
        or source.get("status") != "complete"
        or not isinstance(source_input, Mapping)
        or source_input.get("sample_unit_key") != "subject_id"
        or not isinstance(source_resource, Mapping)
    ):
        raise ValueError("source must be one complete subject-level scSeqCommDiff run")
    dataset_id = str(source.get("dataset_id", ""))
    if not dataset_id or dataset_id != dataset_id.strip():
        raise ValueError("source scSeqCommDiff dataset_id is invalid")
    axes_path = _bound_output(
        source_manifest_path.parent,
        source,
        "condition_cell_pair_rankings.tsv",
    )
    rds_path = _bound_output(
        source_manifest_path.parent,
        source,
        "differential_comm.rds",
    )
    axes = pd.read_csv(axes_path, sep="\t")
    if _single_text(axes, "dataset") != dataset_id:
        raise ValueError("scSeqCommDiff ranking dataset differs from its manifest")

    resource, resource_manifest, resource_provenance = _load_resource(
        resource_manifest_path
    )
    payload_record = resource_provenance["payload"]
    manifest_record = resource_provenance["manifest"]
    if (
        source_resource.get("sha256") != payload_record["sha256"]
        or source_resource.get("manifest_sha256") != manifest_record["sha256"]
    ):
        raise ValueError("scSeqCommDiff source resource differs from the supplied pin")
    resource_id = str(source_resource.get("resource_id", ""))
    if resource_id != resource_manifest.get("resource_id"):
        raise ValueError("scSeqCommDiff resource ID differs from its manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    started = time.perf_counter()
    try:
        raw_path = staged / "scseqcommdiff_event_ledger.tsv.gz"
        exporter = Path(__file__).with_name("export_scseqcommdiff_event_ledger.R")
        raw, export_command, export_elapsed = _export_scseq_ledger(
            rscript=rscript,
            exporter=exporter,
            source_rds=rds_path,
            source_manifest=source,
            output_path=raw_path,
        )
        ledger = prepare_scseqcommdiff_event_ledger(
            raw,
            mechanism_annotations(resource),
        )
        rankings, diagnostics = build_subject_event_track_rankings(
            ledger,
            axes,
            dataset_id=dataset_id,
            resource_id=resource_id,
        )
        ranking_path = staged / "condition_cell_pair_rankings.tsv"
        ledger_path = staged / "scseqcommdiff_event_ledger.tsv.gz"
        diagnostics_path = staged / "track_diagnostics.tsv"
        rankings.to_csv(ranking_path, sep="\t", index=False)
        ledger.to_csv(ledger_path, sep="\t", index=False, compression="gzip")
        diagnostics.to_csv(diagnostics_path, sep="\t", index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset_id": dataset_id,
            "method": {"id": "scseqcommdiff", "version": "2.0.0"},
            "analysis_unit": {
                "replicate_key": "subject_id",
                "subject_key": "subject_id",
                "primary_panel": True,
            },
            "inputs": {
                "h5ad": {"sha256": source_input.get("sha256")},
                "manifest": {
                    "sha256": source_input.get("manifest_sha256")
                },
                "resource_manifest": manifest_record,
            },
            "resource": {
                "id": resource_id,
                "sha256": payload_record["sha256"],
                "rows": len(resource),
            },
            "source_run": _input_record(source_manifest_path),
            "source_outputs": {
                "native_pair_rankings": _input_record(axes_path),
                "differential_comm_rds": _input_record(rds_path),
            },
            "protocol": {
                "top_k_scope": TOP_K_SCOPE,
                "event_budgets": list(EVENT_BUDGETS),
                "top_k_evidence": "negative_log10_native_p",
                "continuous_weight": (
                    "absolute_native_intercellular_score_difference"
                ),
                "formal_p_or_q_emitted": False,
            },
            "export": {
                "command": export_command,
                "elapsed_seconds": export_elapsed,
                "exporter": _input_record(exporter),
            },
            "code": code,
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": {
                ranking_path.name: _output_record(
                    ranking_path, rows=len(rankings)
                ),
                ledger_path.name: _output_record(ledger_path, rows=len(ledger)),
                diagnostics_path.name: _output_record(
                    diagnostics_path, rows=len(diagnostics)
                ),
                raw_path.name: _output_record(raw_path, rows=len(raw)),
            },
        }
        write_json(staged / "manifest.json", json_safe(manifest))
        staged.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = run(
        arguments.source_manifest.resolve(),
        arguments.resource_manifest.resolve(),
        arguments.rscript.resolve(),
        arguments.output_dir.resolve(),
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
