"""Add standardized spatial-DES tables to a completed Kuppe CRYCHIC run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.adapters.crychic.des_postprocess import (
    stable_breadth_unordered_cell_pair_rankings,
    subject_equal_directed_lr_effects,
    unordered_cell_pair_des_rankings,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    DATASET_ID,
    DIRECTED_EDGE_COLUMNS,
    DIRECTED_EFFECT_FILENAME,
    REFERENCE,
    RUN_SCHEMA,
    SCORE_FILENAME,
    SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME,
    TARGET,
    UNORDERED_RANKING_FILENAME,
)

POSTPROCESS_SCHEMA = "crychic-kuppe-spatial-des-postprocess-v1"
SUPPORTED_RUN_SCHEMAS = {
    "crychic-kuppe-ctrl-iz-crossfit-run-v2",
    RUN_SCHEMA,
}


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Kuppe run manifest is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Kuppe run manifest must contain an object")
    return cast(dict[str, Any], value)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": sha256_file(path),
        "columns": list(table.columns),
    }


def run(run_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Validate one completed run, write standard tables, and bind its manifest."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_manifest(manifest_path)
    if (
        manifest.get("schema_version") not in SUPPORTED_RUN_SCHEMAS
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Kuppe run must be complete and use the expected schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(
        outputs.get(SCORE_FILENAME), Mapping
    ):
        raise ValueError("Kuppe run manifest does not bind sender LR scores")
    score_record = cast(Mapping[str, object], outputs[SCORE_FILENAME])
    score_path = root / SCORE_FILENAME
    if not score_path.is_file() or score_record.get("filename") != SCORE_FILENAME:
        raise FileNotFoundError(score_path)
    if score_record.get("sha256") != sha256_file(score_path):
        raise ValueError("sender LR score checksum disagrees with the run manifest")
    scores = pd.read_parquet(score_path)
    if score_record.get("rows") != len(scores):
        raise ValueError("sender LR score row count disagrees with the run manifest")

    method = manifest.get("method")
    resource = manifest.get("resource")
    if not isinstance(method, Mapping) or not isinstance(resource, Mapping):
        raise ValueError("Kuppe run method or resource provenance is missing")
    method_version = _required_text(method.get("version"), field="method.version")
    resource_id = _required_text(
        resource.get("resource_id"), field="resource.resource_id"
    )

    directed_path = root / DIRECTED_EFFECT_FILENAME
    ranking_path = root / UNORDERED_RANKING_FILENAME
    conflicts = [path.name for path in (directed_path, ranking_path) if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {conflicts}")

    directed = subject_equal_directed_lr_effects(
        scores,
        reference=REFERENCE,
        target=TARGET,
        condition_column="condition",
        edge_columns=DIRECTED_EDGE_COLUMNS,
        min_subjects_per_condition=3,
    )
    direct_record = outputs.get(SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME)
    if isinstance(direct_record, Mapping):
        direct_path = root / SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME
        if (
            not direct_path.is_file()
            or direct_record.get("filename")
            != SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME
            or direct_record.get("sha256") != sha256_file(direct_path)
        ):
            raise ValueError(
                "sender-specific directed effects disagree with the run manifest"
            )
        direct_effects = pd.read_parquet(direct_path)
        if direct_record.get("rows") != len(direct_effects):
            raise ValueError(
                "sender-specific directed effect rows disagree with the run manifest"
            )
        rankings, _ = stable_breadth_unordered_cell_pair_rankings(
            direct_effects,
            dataset=DATASET_ID,
            method="crychic",
            method_version=method_version,
            resource=resource_id,
        )
        ranking_source = "persisted_sender_specific_direct_effects"
    else:
        rankings = unordered_cell_pair_des_rankings(
            directed,
            dataset=DATASET_ID,
            method="crychic",
            method_version=method_version,
            resource=resource_id,
        )
        ranking_source = "legacy_global_sender_lr_scores"
    directed.to_parquet(directed_path, index=False, compression="zstd")
    rankings.to_csv(ranking_path, sep="\t", index=False, lineterminator="\n")

    outputs[DIRECTED_EFFECT_FILENAME] = _output_record(directed_path, directed)
    outputs[UNORDERED_RANKING_FILENAME] = _output_record(ranking_path, rankings)
    semantics = manifest.get("score_semantics")
    if not isinstance(semantics, dict):
        semantics = {}
        manifest["score_semantics"] = semantics
    semantics["primary_spatial_des_ranking"] = str(
        rankings["ranking_semantics"].iloc[0]
    )
    manifest["spatial_des_postprocess"] = {
        "schema_version": POSTPROCESS_SCHEMA,
        "input_sender_lr_scores_sha256": score_record["sha256"],
        "statistical_unit": "subject_id",
        "technical_sample_policy": "mean_within_subject_and_condition",
        "minimum_subjects_per_condition": 3,
        "full_model_refit": False,
        "ranking_source": ranking_source,
    }
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = run(args.run_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_dir": str(args.run_dir.resolve()),
                "postprocess": manifest["spatial_des_postprocess"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
