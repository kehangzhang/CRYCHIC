"""Evaluate the frozen RC1 signed-cardinality head on Kuppe or MS once."""

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
from benchmarks.literature.component_swap_benchmark import (
    SCHEMA_VERSION as COMPONENT_SCHEMA_VERSION,
)
from benchmarks.literature.component_swap_benchmark import (
    _arm_diagnostics,
    _build_pair_rankings,
    _evaluate,
    _pair_universe,
    _pairwise_diagnostics,
)
from benchmarks.literature.signed_expected_cardinality import (
    build_signed_expected_cardinality_head,
)

SCHEMA_VERSION = "crychic-frozen-signed-cardinality-real-v1"
METHOD = "CRYCHIC_RC1_soft"
METHOD_VERSION = "signed-cardinality-rc1-v2"
FROZEN_CANDIDATE_SHA256 = (
    "7f2b827d54d1f0c482fcd82d3ac62f1a432d8f8a96948aaa7a66b2217120c844"
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def _validate_frozen_candidate(path: Path) -> dict[str, Any]:
    if sha256_file(path) != FROZEN_CANDIDATE_SHA256:
        raise ValueError("frozen signed-cardinality candidate checksum mismatch")
    candidate = _read_json(path)
    if (
        candidate.get("selected_candidate") != "eb_delta_0.5"
        or candidate.get("selected_on") != "development_only"
        or candidate.get("holdout_used_for_selection") is not False
        or candidate.get("formal_release_allowed") is not False
        or candidate.get("probability_status") != "candidate_unreleased"
    ):
        raise ValueError("frozen signed-cardinality candidate contract mismatch")
    selection = candidate.get("development_selection_diagnostics")
    if (
        not isinstance(selection, Mapping)
        or selection.get("null_gate_pass") is not True
    ):
        raise ValueError("frozen candidate did not pass its development null gate")
    working_prior = candidate.get("working_prior")
    if not isinstance(working_prior, Mapping):
        raise ValueError("frozen candidate lacks working-prior configuration")
    if float(working_prior.get("min_slab_scale_fraction", -1.0)) != 1.5:
        raise ValueError("frozen candidate slab-identifiability bound changed")
    return candidate


def _bound_component_output(
    run: Path, manifest: Mapping[str, Any], filename: str
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(filename), Mapping
    ):
        raise ValueError(f"component run lacks bound output: {filename}")
    record = cast(Mapping[str, Any], outputs[filename])
    path = run / filename
    if record.get("filename") != filename or record.get("sha256") != sha256_file(path):
        raise ValueError(f"component output checksum mismatch: {filename}")
    return path


def _load_component_run(
    run: Path, *, dataset: str, expected_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest_path = run / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != COMPONENT_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
    ):
        raise ValueError("component run schema, status, or dataset mismatch")
    expected = manifest.get("inputs", {}).get("expected_sets")
    if not isinstance(expected, Mapping):
        raise ValueError("component run lacks expected-set provenance")
    if expected.get("sha256") != sha256_file(expected_path):
        raise ValueError("expected-set checksum differs from component run")
    edge_path = _bound_component_output(
        run, manifest, "directed_edge_statistics.parquet"
    )
    ranking_path = _bound_component_output(
        run, manifest, "condition_cell_pair_rankings.tsv"
    )
    edges = pd.read_parquet(edge_path)
    rankings = pd.read_csv(ranking_path, sep="\t")
    provenance = {
        "manifest": _input_record(manifest_path),
        "directed_edges": _input_record(edge_path),
        "rankings": _input_record(ranking_path),
        "expected_sets": _input_record(expected_path),
    }
    return edges, rankings, manifest, provenance


def _component_arm_a_effects(edges: pd.DataFrame) -> pd.DataFrame:
    arm = edges.loc[edges["arm"].eq("A")].copy()
    required = {
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "reference_condition",
        "target_condition",
        "effect",
        "standard_error",
        "n_reference",
        "n_target",
        "status",
        "reason_code",
    }
    missing = required.difference(arm.columns)
    if missing or arm.empty:
        raise ValueError(f"component Arm A is incomplete: {sorted(missing)}")
    return arm.rename(
        columns={
            "effect": "effect_target_minus_reference",
            "standard_error": "effect_standard_error_hc2",
            "n_reference": "n_samples_reference",
            "n_target": "n_samples_target",
        }
    )


def run(
    *,
    dataset: str,
    component_run: Path,
    frozen_candidate_path: Path,
    expected_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if dataset not in {"kuppe", "ms"}:
        raise ValueError("dataset must be 'kuppe' or 'ms'")
    started = time.perf_counter()
    output = prepare_output(output_dir, overwrite=overwrite)
    candidate = _validate_frozen_candidate(frozen_candidate_path)
    component_edges, component_rankings, component_manifest, provenance = (
        _load_component_run(
            component_run.resolve(),
            dataset=dataset,
            expected_path=expected_path.resolve(),
        )
    )
    effects = _component_arm_a_effects(component_edges)
    candidate_spec = cast(Mapping[str, Any], candidate["candidate"])
    working_prior = cast(Mapping[str, Any], candidate["working_prior"])
    dataset_id = str(component_manifest["dataset_id"])
    edge_head, fit = build_signed_expected_cardinality_head(
        effects,
        dataset_id=dataset_id,
        arm=METHOD,
        delta_fraction=float(candidate_spec["delta_fraction"]),
        min_slab_scale_fraction=float(
            working_prior["min_slab_scale_fraction"]
        ),
    )
    pair_rankings = _build_pair_rankings(
        edge_head,
        _pair_universe(effects),
        dataset_id=dataset_id,
        method=METHOD,
        semantics=str(edge_head["selection_rule"].iloc[0]),
    )
    pair_rankings["method_version"] = METHOD_VERSION
    all_rankings = pd.concat([component_rankings, pair_rankings], ignore_index=True)
    expected = pd.read_csv(expected_path, sep="\t")
    scores, coverage, summary = _evaluate(all_rankings, expected, dataset=dataset)
    diagnostics = _arm_diagnostics({METHOD: edge_head}, pair_rankings)
    arm_a = component_edges.loc[component_edges["arm"].eq("A")].copy()
    pairwise = _pairwise_diagnostics(
        {"A": arm_a, METHOD: edge_head},
        pd.concat(
            [
                component_rankings.loc[component_rankings["method"].eq("A")],
                pair_rankings,
            ],
            ignore_index=True,
        ),
    )
    availability = pd.DataFrame.from_records(
        [
            {
                "method": METHOD,
                "status": "complete",
                "backbone": "frozen_component_arm_A",
                "head": "zero_spike_symmetric_normal_slab_expected_count",
                "minimum_effect": (
                    f"0.5*fitted_slab_sd={0.5 * fit.slab_sd:.12g}"
                ),
                "probability_status": "candidate_unreleased",
                "formal_release_allowed": False,
                "reason_code": "not_full_pipeline_calibrated",
            }
        ]
    )

    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "directed_edge_working_probabilities.parquet": (
            output / "directed_edge_working_probabilities.parquet",
            edge_head,
        ),
        "condition_cell_pair_rankings.tsv": (
            output / "condition_cell_pair_rankings.tsv",
            all_rankings,
        ),
        "spatial_des_scores.tsv": (output / "spatial_des_scores.tsv", scores),
        "spatial_des_coverage.tsv": (
            output / "spatial_des_coverage.tsv",
            coverage,
        ),
        "spatial_des_summary.tsv": (
            output / "spatial_des_summary.tsv",
            summary,
        ),
        "arm_diagnostics.tsv": (output / "arm_diagnostics.tsv", diagnostics),
        "pairwise_diagnostics.tsv": (output / "pairwise_diagnostics.tsv", pairwise),
        "arm_availability.tsv": (output / "arm_availability.tsv", availability),
    }
    for filename, (path, table) in paths.items():
        if filename.endswith(".parquet"):
            table.to_parquet(path, index=False, compression="zstd")
        else:
            table.to_csv(path, sep="\t", index=False, lineterminator="\n")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "dataset_id": dataset_id,
        "backbone_modified": False,
        "benchmark_head_modified": True,
        "selection_data": "simulation_development_only",
        "real_dataset_used_once_after_freeze": True,
        "working_probability_status": "candidate_unreleased",
        "formal_release_allowed": False,
        "working_prior_fit": fit.to_dict(),
        "inputs": {
            "component_run": provenance,
            "frozen_candidate": _input_record(frozen_candidate_path),
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "code": git_metadata(Path(__file__).resolve().parents[2]),
        "outputs": {
            filename: _output_record(path, table)
            for filename, (path, table) in paths.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kuppe", "ms"), required=True)
    parser.add_argument("--component-run", type=Path, required=True)
    parser.add_argument("--frozen-candidate", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        component_run=args.component_run,
        frozen_candidate_path=args.frozen_candidate,
        expected_path=args.expected,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
