"""Run frozen RC9 bounded receiver-program blend on Kuppe or MS."""

from __future__ import annotations

import argparse
import json
import resource as process_resource
import time
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
from benchmarks.literature.adaptive_program_gate import (
    adaptive_program_gate_weights,
    fit_adaptive_program_gate,
)
from benchmarks.literature.bounded_program_blend import (
    bounded_program_blend_weights,
)
from benchmarks.literature.component_swap_benchmark import _evaluate
from benchmarks.literature.liana_hypergraph_residual import stable_edge_folds
from benchmarks.literature.run_receiver_program_soft_real import (
    FROZEN_SHA256 as RC3_FROZEN_SHA256,
)
from benchmarks.literature.run_receiver_program_soft_real import (
    METHOD_FALLBACK as RC3_METHOD_FALLBACK,
)
from benchmarks.literature.run_receiver_program_soft_real import (
    METHOD_STRICT as RC3_METHOD_STRICT,
)
from benchmarks.literature.run_receiver_program_soft_real import (
    SCHEMA_VERSION as RC3_SCHEMA_VERSION,
)
from benchmarks.literature.run_receiver_program_soft_real import (
    _bound_output,
    _input_record,
    _output_record,
    _read_json,
    _weighted_strict_rankings,
)
from benchmarks.literature.run_two_part_occurrence_real import _assert_rank_parity

SCHEMA_VERSION = "crychic-bounded-program-blend-real-v1"
METHOD_STRICT = "CRYCHIC_RC9_bounded_program_blend_strict"
METHOD_FALLBACK = "CRYCHIC_RC9_bounded_program_blend_fallback"
METHOD_VERSION = "bounded-program-blend-rc9-v1"
CONFIG_SHA256 = "e597927950dd729a768af70ff43601e0c8f77a2b566d621a51d470cabeaefa53"
ACCEPTANCE_SHA256 = "63ad7c639d3d4bf832a3146d6462c6d019dd5e56899ff6478dd22c217129745a"
FROZEN_SHA256 = "94ee6bf1fa95aa6a22804c64b06f2741dd806bb15556f9b2a92d293c6cee956d"
RC3_SHA256 = {
    "kuppe": {
        "directed_edge_program_weights.parquet": (
            "739634cac2a49aad71a05413799555faae5cae4b00b39547418ee71500ee0026"
        ),
        "condition_cell_pair_rankings.tsv": (
            "78ef42d18063c01f18d554c1200d28116ede89c210a36f0694c60e3ed2990ea6"
        ),
    },
    "ms": {
        "directed_edge_program_weights.parquet": (
            "e275f7ab4587c7e5ac911500d2321758b2e96c1e6eff6cd13862aab1f21113fa"
        ),
        "condition_cell_pair_rankings.tsv": (
            "cc3388a40ab7288f99df909d821e868e86a76f8a6a22995f1d4e514d748892d2"
        ),
    },
}


def _validate_frozen_candidate(
    config_path: Path, acceptance_path: Path, frozen_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        config_path: CONFIG_SHA256,
        acceptance_path: ACCEPTANCE_SHA256,
        frozen_path: FROZEN_SHA256,
    }
    for path, checksum in expected.items():
        if sha256_file(path) != checksum:
            raise ValueError(f"frozen RC9 input checksum mismatch: {path.name}")
    config = _read_json(config_path)
    acceptance = _read_json(acceptance_path)
    frozen = _read_json(frozen_path)
    if acceptance.get("accepted") is not True:
        raise ValueError("RC9 simulation candidate did not pass acceptance")
    if frozen.get("accepted_for_real_benchmark") is not True:
        raise ValueError("RC9 frozen candidate is not accepted for real benchmarking")
    if acceptance.get("selected_candidate") != "blend_r050_e020":
        raise ValueError("unexpected RC9 accepted candidate")
    if frozen.get("selected_candidate") != "blend_r050_e020":
        raise ValueError("unexpected RC9 frozen candidate")
    return config, frozen


def _load_rc3(
    dataset: str, rc3_run: Path, expected_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], str, str]:
    manifest_path = rc3_run / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RC3_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
    ):
        raise ValueError("RC3 run schema, status, or dataset mismatch")
    frozen_record = cast(Mapping[str, Any], manifest.get("inputs", {})).get(
        "frozen_candidate"
    )
    if not isinstance(frozen_record, Mapping) or frozen_record.get(
        "sha256"
    ) != RC3_FROZEN_SHA256:
        raise ValueError("RC3 source does not bind the frozen RC3 candidate")
    edge_path = _bound_output(
        rc3_run,
        manifest,
        "directed_edge_program_weights.parquet",
        expected_sha256=RC3_SHA256[dataset][
            "directed_edge_program_weights.parquet"
        ],
    )
    ranking_path = _bound_output(
        rc3_run,
        manifest,
        "condition_cell_pair_rankings.tsv",
        expected_sha256=RC3_SHA256[dataset]["condition_cell_pair_rankings.tsv"],
    )
    expected_record = (
        cast(Mapping[str, Any], manifest.get("inputs", {}))
        .get("rc2", {})
        .get("expected_sets")
    )
    if not isinstance(expected_record, Mapping) or expected_record.get(
        "sha256"
    ) != sha256_file(expected_path):
        raise ValueError("expected spatial sets do not match the frozen RC3 run")
    contrast = manifest.get("contrast")
    if not isinstance(contrast, Mapping):
        raise ValueError("RC3 source contrast is missing")
    edges = pd.read_parquet(edge_path)
    rankings = pd.read_csv(ranking_path, sep="\t")
    required = {
        "sender",
        "receiver",
        "interaction_id",
        "interaction_pvalue",
        "final_sign_statistic",
        "program_z",
        "program_weight",
        "selected_rc2_call",
        "formal_inference_allowed",
    }
    if required.difference(edges.columns):
        raise ValueError("RC3 edge source is incomplete")
    if edges["formal_inference_allowed"].fillna(True).astype(bool).any():
        raise ValueError("RC3 formal release contract was altered")
    provenance = {
        "manifest": _input_record(manifest_path),
        "edges": _input_record(edge_path),
        "rankings": _input_record(ranking_path),
        "expected_sets": _input_record(expected_path),
    }
    return (
        edges,
        rankings,
        pd.read_csv(expected_path, sep="\t"),
        provenance,
        str(contrast["target"]),
        str(contrast["reference"]),
    )


def _coverage_fallback(
    strict: pd.DataFrame, all_rc3_rankings: pd.DataFrame
) -> pd.DataFrame:
    result = strict.copy()
    observed = result["status"].eq("observed")
    result["_strict_rank"] = np.nan
    result.loc[observed, "_strict_rank"] = (
        result.loc[observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    source = all_rc3_rankings.loc[
        all_rc3_rankings["method"].eq(RC3_METHOD_FALLBACK),
        [
            "condition",
            "sender",
            "receiver",
            "ranked_strength",
            "estimable_directed_lr",
            "status",
        ],
    ].rename(
        columns={
            "ranked_strength": "_fallback_rank",
            "estimable_directed_lr": "_fallback_estimable",
            "status": "_fallback_status",
        }
    )
    result = result.merge(
        source,
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    use_fallback = ~observed & result["_fallback_status"].eq("observed")
    result["ranked_strength"] = result["_strict_rank"].where(
        observed, result["_fallback_rank"]
    )
    result.loc[use_fallback, "estimable_directed_lr"] = result.loc[
        use_fallback, "_fallback_estimable"
    ]
    result.loc[use_fallback, "condition_specific_directed_lr"] = np.nan
    result["status"] = np.where(
        result["ranked_strength"].notna(), "observed", "not_estimable"
    )
    result["reason_code"] = np.where(
        observed,
        None,
        np.where(
            use_fallback,
            "bounded_program_not_estimable_used_rc3_fallback",
            "bounded_program_and_rc3_fallback_not_estimable",
        ),
    )
    result["method"] = METHOD_FALLBACK
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = (
        "bounded_soft_gate_cardinality_percentile_with_rc3_coverage_fallback"
    )
    return result.drop(
        columns=[
            "_strict_rank",
            "_fallback_rank",
            "_fallback_estimable",
            "_fallback_status",
        ]
    )


def run(
    *,
    dataset: str,
    rc3_run: Path,
    expected_path: Path,
    config_path: Path,
    acceptance_path: Path,
    frozen_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, frozen = _validate_frozen_candidate(
        config_path, acceptance_path, frozen_path
    )
    output = prepare_output(output_dir, overwrite=overwrite)
    edges, rc3_rankings, expected, provenance, target, reference = _load_rc3(
        dataset, rc3_run.resolve(), expected_path.resolve()
    )
    strict_template = rc3_rankings.loc[
        rc3_rankings["method"].eq(RC3_METHOD_STRICT)
    ].copy()
    if strict_template.empty:
        raise ValueError("RC3 strict ranking template is missing")
    stored_call = edges["selected_rc2_call"].astype(bool).to_numpy()
    recomputed_call = (
        pd.to_numeric(edges["interaction_pvalue"], errors="coerce").lt(0.05)
        & pd.to_numeric(edges["final_sign_statistic"], errors="coerce").ne(0.0)
    ).to_numpy()
    np.testing.assert_array_equal(stored_call, recomputed_call)
    parity_rankings, parity_calls = _weighted_strict_rankings(
        edges,
        strict_template,
        edges["program_weight"].to_numpy(dtype=float),
        target=target,
        reference=reference,
        method="RC9_rc3_parity_check",
        semantics="rc3_program_weight_parity_check",
    )
    parity = _assert_rank_parity(parity_rankings, strict_template)
    parity["selected_calls"] = parity_calls

    candidate = cast(Mapping[str, Any], frozen["candidate"])
    gate_policy = cast(Mapping[str, Any], frozen["gate_policy"])
    sign = pd.to_numeric(edges["final_sign_statistic"], errors="coerce").to_numpy(
        dtype=float
    )
    direction = np.where(sign >= 0.0, 1.0, -1.0)
    validation_fold = stable_edge_folds(
        edges,
        key_columns=tuple(str(value) for value in config["residual"]["key_columns"]),
        folds=int(gate_policy["validation_folds"]),
        seed=int(gate_policy["validation_seed"]),
    )
    gate_fit = fit_adaptive_program_gate(
        direction,
        edges["program_z"].to_numpy(dtype=float),
        stored_call,
        relative_fdp_reduction=float(candidate["relative_fdp_reduction"]),
        minimum_calls=int(gate_policy["minimum_calls"]),
        minimum_sign_concordance=float(gate_policy["minimum_sign_concordance"]),
        minimum_positive_tail_count=int(
            gate_policy["minimum_positive_tail_count"]
        ),
        minimum_positive_tail_fraction=float(
            gate_policy["minimum_positive_tail_fraction"]
        ),
        tail_confidence_z=float(gate_policy["tail_confidence_z"]),
        minimum_tail_concordance_margin=float(
            gate_policy["minimum_tail_concordance_margin"]
        ),
        validation_fold=validation_fold,
        minimum_validation_tail_count=int(
            gate_policy["minimum_validation_tail_count"]
        ),
        minimum_validation_concordance_gain=float(
            gate_policy["minimum_validation_concordance_gain"]
        ),
    )
    gate_weights = adaptive_program_gate_weights(
        direction, edges["program_z"].to_numpy(dtype=float), gate_fit
    )
    blended, blend_fit = bounded_program_blend_weights(
        edges["program_weight"].to_numpy(dtype=float),
        gate_weights,
        stored_call,
        blend_fraction=float(candidate["blend_fraction"]),
    )
    strict, selected_calls = _weighted_strict_rankings(
        edges,
        strict_template,
        blended,
        target=target,
        reference=reference,
        method=METHOD_STRICT,
        semantics=(
            "exact_rc3_calls_weighted_by_mean_preserving_bounded_soft_and_"
            "mirror_tail_program_experts"
        ),
    )
    if selected_calls != parity_calls:
        raise AssertionError("RC9 changed the frozen RC3 call set")
    strict["method_version"] = METHOD_VERSION
    fallback = _coverage_fallback(strict, rc3_rankings)
    all_rankings = pd.concat([rc3_rankings, strict, fallback], ignore_index=True)
    scores, coverage, summary = _evaluate(all_rankings, expected, dataset=dataset)

    edge_output = edges.copy()
    edge_output["rc9_gate_weight"] = gate_weights
    edge_output["rc9_blended_weight"] = blended
    edge_output["rc9_validation_fold"] = validation_fold
    edge_output["formal_inference_allowed"] = False
    fit_diagnostics = pd.DataFrame(
        [
            {
                "dataset": dataset,
                **{f"gate_{key}": value for key, value in gate_fit.to_dict().items()},
                **{
                    f"blend_{key}": value for key, value in blend_fit.to_dict().items()
                },
            }
        ]
    )
    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "directed_edge_bounded_program_weights.parquet": (
            output / "directed_edge_bounded_program_weights.parquet",
            edge_output,
        ),
        "fit_diagnostics.tsv": (output / "fit_diagnostics.tsv", fit_diagnostics),
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
    }
    for filename, (path, table) in paths.items():
        if filename.endswith(".parquet"):
            table.to_parquet(path, index=False, compression="zstd")
        else:
            table.to_csv(path, sep="\t", index=False, lineterminator="\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "contrast": {"target": target, "reference": reference},
        "frozen_candidate": frozen,
        "rc3_parity": parity,
        "gate_fit": gate_fit.to_dict(),
        "blend_fit": blend_fit.to_dict(),
        "alignment": {
            "rc3_edges": len(edges),
            "selected_rc3_calls": selected_calls,
            "rc3_call_set_modified": False,
            "missing_program_evidence_is_neutral": True,
        },
        "backbone_modified": False,
        "benchmark_head_modified": True,
        "formal_release_allowed": False,
        "limitations": [
            "receiver_program_source_is_partial_pipeline_crossfit_evidence",
            "binary_gate_is_used_only_as_a_bounded_benchmark_expert",
            "spatial_colocalization_is_an_indirect_proxy",
        ],
        "inputs": {
            "rc3": provenance,
            "config": _input_record(config_path),
            "acceptance": _input_record(acceptance_path),
            "frozen_candidate": _input_record(frozen_path),
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
    parser.add_argument("--rc3-run", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        rc3_run=args.rc3_run,
        expected_path=args.expected,
        config_path=args.config,
        acceptance_path=args.acceptance,
        frozen_path=args.frozen,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "gate_fit": result["gate_fit"],
                "blend_fit": result["blend_fit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
