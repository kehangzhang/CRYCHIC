"""Replay the exploratory V3 multigroup differential head from persisted runs."""

from __future__ import annotations

import argparse
import json
import resource as process_resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.des_postprocess import (
    adjusted_subject_directed_lr_effects,
    condition_ranking_diagnostics,
    score_reason_waterfall,
    sender_specific_direct_scores,
    stable_breadth_unordered_cell_pair_rankings,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    DATASET_ID as KUPPE_DATASET_ID,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    DIRECTED_EDGE_COLUMNS as KUPPE_EDGE_COLUMNS,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    MECHANISTIC_DIRECTED_EFFECT_FILENAME as KUPPE_MECHANISTIC_EFFECT_FILENAME,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    REFERENCE as KUPPE_REFERENCE,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import (
    SCORE_LAYER_FILENAME as KUPPE_SCORE_LAYER_FILENAME,
)
from benchmarks.adapters.crychic.run_kuppe_ctrl_iz import TARGET as KUPPE_TARGET
from benchmarks.adapters.crychic.run_ms_ctrl_ca import DES_DATASET_ID as MS_DATASET_ID
from benchmarks.adapters.crychic.run_ms_ctrl_ca import (
    DIRECTED_EDGE_COLUMNS as MS_EDGE_COLUMNS,
)
from benchmarks.adapters.crychic.run_ms_ctrl_ca import (
    MECHANISTIC_DIRECTED_EFFECT_FILENAME as MS_MECHANISTIC_EFFECT_FILENAME,
)
from benchmarks.adapters.crychic.run_ms_ctrl_ca import REFERENCE as MS_REFERENCE
from benchmarks.adapters.crychic.run_ms_ctrl_ca import (
    SCORE_LAYER_FILENAME as MS_SCORE_LAYER_FILENAME,
)
from benchmarks.adapters.crychic.run_ms_ctrl_ca import TARGET as MS_TARGET

SCHEMA_VERSION = "crychic-multigroup-v3-differential-replay-v1"
DIRECT_EFFECT_FILENAME = "sender_specific_directed_lr_effects.parquet"
RANKING_FILENAME = "condition_cell_pair_rankings.tsv"
PAIR_OPPORTUNITY_FILENAME = "pair_opportunity.tsv"
REASON_WATERFALL_FILENAME = "downstream_reason_waterfall.tsv"
EDGE_COMPONENT_DELTA_FILENAME = "edge_component_delta.parquet"


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _bound_output(
    run_dir: Path,
    manifest: Mapping[str, object],
    filename: str,
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(filename), Mapping
    ):
        raise ValueError(f"source run does not bind {filename}")
    record = cast(Mapping[str, object], outputs[filename])
    path = run_dir / filename
    if (
        not path.is_file()
        or record.get("filename") != filename
        or record.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"source run binding is invalid for {filename}")
    return path


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def _validated_crossfit_semantic_source(
    source: Path,
    source_manifest: Mapping[str, object],
) -> tuple[Path, Path, str, str]:
    binding = source_manifest.get("crossfit_result")
    if not isinstance(binding, Mapping):
        raise ValueError("source run lacks a cross-fit result binding")
    directory = binding.get("directory")
    if not isinstance(directory, str) or Path(directory).name != directory:
        raise ValueError("source run has an invalid cross-fit directory binding")
    crossfit_dir = (source / directory).resolve()
    if crossfit_dir.parent != source:
        raise ValueError("cross-fit directory escapes the source run")
    crossfit_manifest_path = crossfit_dir / "crossfit_manifest.json"
    if not crossfit_manifest_path.is_file():
        raise FileNotFoundError(crossfit_manifest_path)
    crossfit_manifest_sha = sha256_file(crossfit_manifest_path)
    if binding.get("manifest_sha256") != crossfit_manifest_sha:
        raise ValueError("cross-fit manifest checksum disagrees with source run")
    crossfit_manifest = _read_json(crossfit_manifest_path)
    if (
        crossfit_manifest.get("status") != "complete"
        or crossfit_manifest.get("schema_version") != binding.get("schema_version")
        or crossfit_manifest.get("crossfit_result_id")
        != binding.get("crossfit_result_id")
    ):
        raise ValueError("cross-fit manifest identity disagrees with source run")
    tables = crossfit_manifest.get("tables")
    record = (
        tables.get("semantic_availability_scores")
        if isinstance(tables, Mapping)
        else None
    )
    if not isinstance(record, Mapping):
        raise ValueError("cross-fit manifest lacks semantic availability")
    filename = record.get("filename")
    if filename != "semantic_availability_scores.parquet":
        raise ValueError("semantic availability filename is invalid")
    semantic_path = crossfit_dir / filename
    if not semantic_path.is_file():
        raise FileNotFoundError(semantic_path)
    semantic_sha = sha256_file(semantic_path)
    if record.get("sha256") != semantic_sha:
        raise ValueError("semantic availability checksum disagrees with manifest")
    return (
        crossfit_manifest_path,
        semantic_path,
        crossfit_manifest_sha,
        semantic_sha,
    )


def _dataset_contract(
    dataset: str,
) -> tuple[
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
]:
    if dataset == "ms":
        return (
            MS_DATASET_ID,
            MS_REFERENCE,
            MS_TARGET,
            "lesion_type",
            ("batch",),
            tuple(MS_EDGE_COLUMNS),
            MS_SCORE_LAYER_FILENAME,
            MS_MECHANISTIC_EFFECT_FILENAME,
        )
    if dataset == "kuppe":
        return (
            KUPPE_DATASET_ID,
            KUPPE_REFERENCE,
            KUPPE_TARGET,
            "condition",
            (),
            tuple(KUPPE_EDGE_COLUMNS),
            KUPPE_SCORE_LAYER_FILENAME,
            KUPPE_MECHANISTIC_EFFECT_FILENAME,
        )
    raise ValueError("dataset must be 'ms' or 'kuppe'")


def run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay V3 without refitting the cross-fit backbone."""

    started = time.perf_counter()
    source = Path(run_dir).expanduser().resolve()
    output = prepare_output(output_dir, overwrite=overwrite)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source_manifest = _read_json(manifest_path)
    if source_manifest.get("status") != "complete":
        raise ValueError("source run must be complete")
    (
        dataset_id,
        reference,
        target,
        condition_column,
        covariates,
        edge_columns,
        score_layer_filename,
        mechanistic_effect_filename,
    ) = _dataset_contract(dataset)
    score_layer_path = _bound_output(
        source, source_manifest, score_layer_filename
    )
    mechanistic_effect_path = _bound_output(
        source, source_manifest, mechanistic_effect_filename
    )
    (
        _crossfit_manifest_path,
        semantic_path,
        crossfit_manifest_sha,
        semantic_sha,
    ) = _validated_crossfit_semantic_source(source, source_manifest)

    score_columns = list(
        dict.fromkeys(
            (
                "fold_id",
                "sample_id",
                "subject_id",
                condition_column,
                *covariates,
                *edge_columns,
                "mechanistic_status",
                "mechanistic_reason_code",
                "status",
                "reason_code",
                "downstream_status",
                "downstream_reason_code",
                "selected_score_status",
                "selected_score_reason_code",
            )
        )
    )
    score_layers = pd.read_parquet(score_layer_path, columns=score_columns)
    semantic = pd.read_parquet(
        semantic_path,
        columns=[
            "fold_id",
            "sample_id",
            "subject_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "availability_score",
            "status",
            "reason_code",
        ],
        filters=[("mode", "==", "state")],
    )
    direct_scores = sender_specific_direct_scores(
        score_layers,
        semantic,
        condition_column=condition_column,
        covariate_columns=covariates,
        edge_columns=edge_columns,
        response_transform="identity",
    )
    del semantic
    waterfall = score_reason_waterfall(score_layers, direct_scores)
    effects = adjusted_subject_directed_lr_effects(
        direct_scores,
        reference=reference,
        target=target,
        condition_column=condition_column,
        categorical_covariates=covariates,
        edge_columns=edge_columns,
        min_subjects_per_condition=3,
    )
    del direct_scores

    method = source_manifest.get("method")
    resource = source_manifest.get("resource")
    if not isinstance(method, Mapping) or not isinstance(resource, Mapping):
        raise ValueError("source run lacks method/resource provenance")
    code_metadata = git_metadata(Path(__file__).resolve().parents[3])
    source_method_version = str(method.get("version"))
    source_commit = code_metadata.get("commit")
    method_version = (
        source_method_version
        if not isinstance(source_commit, str) or not source_commit
        else f"{source_method_version};v3_source={source_commit[:12]}"
    )
    resource_id = str(resource.get("resource_id"))
    rankings, pair_opportunity = stable_breadth_unordered_cell_pair_rankings(
        effects,
        dataset=dataset_id,
        method="crychic",
        method_version=method_version,
        resource=resource_id,
    )
    ranking_diagnostics = condition_ranking_diagnostics(rankings)
    if ranking_diagnostics["any_degenerate_condition"]:
        raise RuntimeError("V3 replay produced a degenerate condition ranking")

    mechanistic = pd.read_parquet(mechanistic_effect_path)
    direct_component = effects.loc[
        :,
        [
            *edge_columns,
            "mean_strength_reference",
            "mean_strength_target",
            "effect_target_minus_reference",
            "effect_standard_error_hc2",
            "one_standard_error_stable",
            "status",
            "reason_code",
        ],
    ].rename(
        columns={
            column: f"direct_{column}"
            for column in (
                "mean_strength_reference",
                "mean_strength_target",
                "effect_target_minus_reference",
                "effect_standard_error_hc2",
                "one_standard_error_stable",
                "status",
                "reason_code",
            )
        }
    )
    mechanistic_component = mechanistic.loc[
        :,
        [
            *edge_columns,
            "mean_strength_reference",
            "mean_strength_target",
            "effect_target_minus_reference",
            "status",
            "reason_code",
        ],
    ].rename(
        columns={
            column: f"mechanistic_{column}"
            for column in (
                "mean_strength_reference",
                "mean_strength_target",
                "effect_target_minus_reference",
                "status",
                "reason_code",
            )
        }
    )
    edge_component_delta = direct_component.merge(
        mechanistic_component,
        on=list(edge_columns),
        how="outer",
        validate="one_to_one",
        sort=False,
    )

    direct_effect_path = output / DIRECT_EFFECT_FILENAME
    ranking_path = output / RANKING_FILENAME
    opportunity_path = output / PAIR_OPPORTUNITY_FILENAME
    waterfall_path = output / REASON_WATERFALL_FILENAME
    component_path = output / EDGE_COMPONENT_DELTA_FILENAME
    effects.to_parquet(direct_effect_path, index=False, compression="zstd")
    rankings.to_csv(ranking_path, sep="\t", index=False, lineterminator="\n")
    pair_opportunity.to_csv(
        opportunity_path, sep="\t", index=False, lineterminator="\n"
    )
    waterfall.to_csv(waterfall_path, sep="\t", index=False, lineterminator="\n")
    edge_component_delta.to_parquet(
        component_path, index=False, compression="zstd"
    )

    observed_effect = effects.loc[
        effects["status"].eq("observed"), "effect_target_minus_reference"
    ]
    result_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "dataset_id": dataset_id,
        "reference": reference,
        "target": target,
        "release_status": "exploratory_v3",
        "source_run": {
            "path": str(source),
            "manifest_sha256": sha256_file(manifest_path),
            "crossfit_manifest_sha256": crossfit_manifest_sha,
            "score_layers_sha256": sha256_file(score_layer_path),
            "semantic_availability_sha256": semantic_sha,
            "mechanistic_effects_sha256": sha256_file(mechanistic_effect_path),
        },
        "method": {
            "id": "crychic",
            "version": method_version,
            "backbone_version": source_method_version,
        },
        "resource": dict(resource),
        "code": code_metadata,
        "implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "des_postprocess_sha256": sha256_file(
                Path(__file__).with_name("des_postprocess.py")
            ),
            "response": "sender_specific_ligand_x_receptor_availability",
            "technical_sample_policy": "mean_within_subject",
            "categorical_covariates": list(covariates),
            "uncertainty": (
                "contrast_specific_hc2_heteroskedasticity_robust_standard_error"
            ),
            "ranking": "one_standard_error_stable_lr_breadth",
            "formal_inference_allowed": False,
        },
        "diagnostics": {
            "condition_rankings": ranking_diagnostics,
            "effect_sign_counts": {
                "positive": int(observed_effect.gt(0.0).sum()),
                "negative": int(observed_effect.lt(0.0).sum()),
                "zero": int(observed_effect.eq(0.0).sum()),
            },
            "observed_effect_rows": len(observed_effect),
            "stable_effect_rows": int(
                effects.loc[
                    effects["status"].eq("observed"),
                    "one_standard_error_stable",
                ].sum()
            ),
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "outputs": {
            DIRECT_EFFECT_FILENAME: _output_record(direct_effect_path, effects),
            RANKING_FILENAME: _output_record(ranking_path, rankings),
            PAIR_OPPORTUNITY_FILENAME: _output_record(
                opportunity_path, pair_opportunity
            ),
            REASON_WATERFALL_FILENAME: _output_record(waterfall_path, waterfall),
            EDGE_COMPONENT_DELTA_FILENAME: _output_record(
                component_path, edge_component_delta
            ),
        },
    }
    write_json(output / "manifest.json", result_manifest)
    return result_manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", choices=("ms", "kuppe"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = run(
        args.run_dir,
        args.output_dir,
        dataset=args.dataset,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset": manifest["dataset"],
                "output": str(args.output_dir.resolve()),
                "performance": manifest["performance"],
                "diagnostics": manifest["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
