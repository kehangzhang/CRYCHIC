"""Run frozen RC4 adaptive residual views on Kuppe or MS."""

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
from benchmarks.literature.adaptive_multiview_residual import (
    estimate_oof_view_weights,
    fit_adaptive_cross_validated_residual,
)
from benchmarks.literature.component_swap_benchmark import _evaluate
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
)
from benchmarks.literature.receiver_program_soft import (
    fit_receiver_program_reliability,
    receiver_program_weights,
)
from benchmarks.literature.run_liana_hypergraph_residual_real import (
    METHOD_STRICT as RC2_METHOD_STRICT,
)
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
    _assert_alpha_zero_parity,
    _input_record,
    _load_program_run,
    _load_rc2,
    _output_record,
    _weighted_strict_rankings,
    derive_receiver_program_effects,
)
from benchmarks.literature.run_receiver_program_soft_real import (
    _coverage_fallback as _rc3_coverage_fallback,
)

SCHEMA_VERSION = "crychic-adaptive-multiview-real-v1"
METHOD_STRICT = "CRYCHIC_RC4_adaptive_residual_strict"
METHOD_FALLBACK = "CRYCHIC_RC4_adaptive_residual_fallback"
METHOD_PROGRAM_STRICT = "CRYCHIC_RC4_adaptive_program_strict"
METHOD_PROGRAM_FALLBACK = "CRYCHIC_RC4_adaptive_program_fallback"
METHOD_VERSION = "adaptive-multiview-residual-rc4-v1"
CONFIG_SHA256 = "e40867a75fa348d9caf4288b33e9df0a671764aba2707a8d414ed78dcaa4a9ff"
ACCEPTANCE_SHA256 = "28e8778efc58db48ad860ee57cf966cb4cd299625e56b6528a1750f2a277187d"


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _validate_frozen_candidate(
    config_path: Path,
    acceptance_path: Path,
    rc3_frozen_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("frozen RC4 config checksum mismatch")
    if sha256_file(acceptance_path) != ACCEPTANCE_SHA256:
        raise ValueError("frozen RC4 acceptance checksum mismatch")
    if sha256_file(rc3_frozen_path) != RC3_FROZEN_SHA256:
        raise ValueError("frozen RC3 program candidate checksum mismatch")
    acceptance = _read_json(acceptance_path)
    if acceptance.get("accepted") is not True:
        raise ValueError("RC4 simulation candidate did not pass acceptance")
    rc3_frozen = _read_json(rc3_frozen_path)
    if rc3_frozen.get("selected_candidate") != "program_max_alpha_4":
        raise ValueError("unexpected frozen RC3 program candidate")
    return _read_json(config_path), rc3_frozen


def _adaptive_profiles(
    edges: pd.DataFrame,
    baseline: np.ndarray,
    residual: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, ...]], pd.DataFrame]:
    policy = cast(Mapping[str, Any], residual["profile_policy"])
    profile_names = tuple(str(value) for value in policy["profiles"])
    if profile_names != ("all_equal", "reliability_weighted"):
        raise ValueError("RC4 frozen profile policy was altered")
    views = tuple(str(value) for value in residual["views"])
    weights, diagnostics = estimate_oof_view_weights(
        edges,
        baseline,
        views=views,
        key_columns=tuple(str(value) for value in residual["key_columns"]),
        folds=int(residual["folds"]),
        seed=int(residual["seed"]),
        full_reliability_improvement=float(policy["full_reliability_improvement"]),
    )
    return {
        "all_equal": tuple(np.ones(len(views), dtype=float)),
        "reliability_weighted": weights,
    }, diagnostics


def _rc4_strict_rankings(
    edges: pd.DataFrame,
    template: pd.DataFrame,
    weights: np.ndarray,
    *,
    target: str,
    reference: str,
    method: str,
    semantics: str,
) -> tuple[pd.DataFrame, int]:
    result, calls = _weighted_strict_rankings(
        edges,
        template,
        weights,
        target=target,
        reference=reference,
        method=method,
        semantics=semantics,
    )
    result["method_version"] = METHOD_VERSION
    return result, calls


def _rc4_coverage_fallback(
    strict: pd.DataFrame,
    all_rc2_rankings: pd.DataFrame,
    *,
    method: str,
    reason_prefix: str,
    semantics: str,
) -> pd.DataFrame:
    result = strict.copy()
    observed = result["status"].eq("observed")
    result["_strict_rank"] = np.nan
    result.loc[observed, "_strict_rank"] = (
        result.loc[observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = all_rc2_rankings.loc[
        all_rc2_rankings["method"].eq("C"),
        [
            "condition",
            "sender",
            "receiver",
            "ranked_strength",
            "estimable_directed_lr",
            "status",
        ],
    ].copy()
    fallback_observed = fallback["status"].eq("observed")
    fallback["_fallback_rank"] = np.nan
    fallback.loc[fallback_observed, "_fallback_rank"] = (
        fallback.loc[fallback_observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = fallback.rename(columns={"estimable_directed_lr": "_fallback_estimable"})
    result = result.merge(
        fallback.loc[
            :,
            [
                "condition",
                "sender",
                "receiver",
                "_fallback_rank",
                "_fallback_estimable",
            ],
        ],
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    observed = result["status"].eq("observed")
    use_fallback = ~observed & result["_fallback_rank"].notna()
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
            f"{reason_prefix}_used_crychic_wald_rank_fallback",
            f"{reason_prefix}_and_crychic_fallback_not_estimable",
        ),
    )
    result["method"] = method
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = semantics
    return result.drop(
        columns=["_strict_rank", "_fallback_rank", "_fallback_estimable"]
    )


def run(
    *,
    dataset: str,
    rc2_run: Path,
    program_run: Path,
    expected_path: Path,
    config_path: Path,
    acceptance_path: Path,
    rc3_frozen_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, rc3_frozen = _validate_frozen_candidate(
        config_path, acceptance_path, rc3_frozen_path
    )
    output = prepare_output(output_dir, overwrite=overwrite)
    edges, rc2_rankings, expected, rc2_provenance = _load_rc2(
        dataset, rc2_run.resolve(), expected_path.resolve()
    )
    (
        program,
        sample_conditions,
        family_map,
        target,
        reference,
        program_provenance,
    ) = _load_program_run(dataset, program_run.resolve())
    strict_template = rc2_rankings.loc[
        rc2_rankings["method"].eq(RC2_METHOD_STRICT)
    ].copy()
    if set(strict_template["condition"].astype(str)) != {target, reference}:
        raise ValueError("RC2 and receiver-program contrasts differ")

    residual = cast(Mapping[str, Any], config["residual"])
    views = tuple(str(value) for value in residual["views"])
    for source, view in zip(
        ("sender", "receiver", "ligand", "receptor", "interaction_id"),
        views,
        strict=True,
    ):
        edges[view] = edges[source].astype(str)
    baseline = edges["interaction_stat"].to_numpy(dtype=float)
    anchor = edges["crychic_z"].to_numpy(dtype=float)
    common = {
        "views": views,
        "key_columns": tuple(str(value) for value in residual["key_columns"]),
        "lambda_grid": tuple(float(value) for value in residual["lambda_grid"]),
        "anchor_grid": tuple(float(value) for value in residual["anchor_grid"]),
        "folds": int(residual["folds"]),
        "seed": int(residual["seed"]),
        "minimum_cv_improvement": float(residual["minimum_cv_improvement"]),
        "full_gate_improvement": float(residual["full_gate_improvement"]),
    }
    equal_theta, equal_fit, _ = fit_cross_validated_residual(
        edges, baseline, anchor, **common
    )
    np.testing.assert_array_equal(
        equal_theta, edges["residual_theta"].to_numpy(dtype=float)
    )
    equal_sign = conservative_sign_statistic(
        baseline,
        equal_theta,
        equal_fit,
        maximum_abs_baseline=float(residual["sign_correction_max_abs_baseline"]),
    )
    np.testing.assert_array_equal(
        equal_sign, edges["final_sign_statistic"].to_numpy(dtype=float)
    )
    parity_rankings, parity_calls = _weighted_strict_rankings(
        edges,
        strict_template,
        np.ones(len(edges), dtype=float),
        target=target,
        reference=reference,
        method="RC4_equal_parity_check",
        semantics="equal_profile_rc2_parity_check",
    )
    parity = _assert_alpha_zero_parity(parity_rankings, strict_template)
    parity["selected_calls"] = parity_calls
    parity["theta_exact"] = True
    parity["sign_exact"] = True

    profiles, view_reliability = _adaptive_profiles(edges, baseline, residual)
    adaptive_theta, adaptive_fit, cv = fit_adaptive_cross_validated_residual(
        edges,
        baseline,
        anchor,
        profiles=profiles,
        minimum_profile_relative_improvement=float(
            residual["minimum_profile_relative_improvement"]
        ),
        **common,
    )
    adaptive_sign = conservative_sign_statistic(
        baseline,
        adaptive_theta,
        adaptive_fit,
        maximum_abs_baseline=float(residual["sign_correction_max_abs_baseline"]),
    )
    adaptive_edges = edges.copy()
    adaptive_edges["rc2_final_sign_statistic"] = adaptive_edges["final_sign_statistic"]
    adaptive_edges["adaptive_residual_theta"] = adaptive_theta
    adaptive_edges["final_sign_statistic"] = adaptive_sign
    adaptive_edges["adaptive_sign_flipped_vs_rc2"] = np.sign(adaptive_sign) != np.sign(
        equal_sign
    )
    adaptive_strict, adaptive_calls = _rc4_strict_rankings(
        adaptive_edges,
        strict_template,
        np.ones(len(adaptive_edges), dtype=float),
        target=target,
        reference=reference,
        method=METHOD_STRICT,
        semantics=(
            "exact_liana_raw_p_lt_0.05_cardinality_with_cv_gated_adaptive_"
            "nonnegative_multiview_residual_direction"
        ),
    )
    if adaptive_calls != parity_calls:
        raise AssertionError("adaptive residual changed the frozen LIANA call set")
    adaptive_fallback = _rc4_coverage_fallback(
        adaptive_strict,
        rc2_rankings,
        method=METHOD_FALLBACK,
        reason_prefix="adaptive_residual_not_estimable",
        semantics=(
            "adaptive_residual_cardinality_percentile_with_crychic_wald_"
            "percentile_only_when_liana_pair_not_estimable"
        ),
    )

    effects = derive_receiver_program_effects(
        program,
        sample_conditions,
        condition_column="condition" if dataset == "kuppe" else "lesion_type",
        target=target,
        reference=reference,
    )
    family_keys = ["sender", "receiver", "ligand", "receptor"]
    adaptive_edges = adaptive_edges.merge(
        family_map,
        on=family_keys,
        how="left",
        validate="one_to_one",
    ).merge(
        effects.loc[:, ["receiver", "family_id", "program_z", "status"]].rename(
            columns={"status": "program_status"}
        ),
        on=["receiver", "family_id"],
        how="left",
        validate="many_to_one",
    )
    rc3_candidate = cast(Mapping[str, Any], rc3_frozen["candidate"])
    rc3_policy = cast(Mapping[str, Any], rc3_frozen["program_policy"])
    program_fit = fit_receiver_program_reliability(
        baseline,
        adaptive_edges["program_z"].to_numpy(dtype=float),
        maximum_alpha=float(rc3_candidate["maximum_alpha"]),
        alpha_cap=float(rc3_policy["alpha_cap"]),
        minimum_edges=int(rc3_policy["minimum_concordance_edges"]),
    )
    rc2_direction = np.where(equal_sign >= 0.0, 1.0, -1.0)
    rc3_weights, rc3_evidence = receiver_program_weights(
        rc2_direction,
        adaptive_edges["program_z"].to_numpy(dtype=float),
        program_fit,
    )
    rc3_strict, rc3_calls = _weighted_strict_rankings(
        edges,
        strict_template,
        rc3_weights,
        target=target,
        reference=reference,
        method=RC3_METHOD_STRICT,
        semantics=(
            "exact_rc2_calls_weighted_by_reliability_shrunk_oof_receiver_program_"
            "evidence;missing_program_evidence_neutral"
        ),
    )
    if rc3_calls != parity_calls:
        raise AssertionError("recreated RC3 changed the frozen RC2 call set")
    rc3_fallback = _rc3_coverage_fallback(rc3_strict, rc2_rankings)
    if not rc3_fallback["method"].eq(RC3_METHOD_FALLBACK).all():
        raise AssertionError("recreated RC3 fallback method contract changed")

    adaptive_direction = np.where(adaptive_sign >= 0.0, 1.0, -1.0)
    adaptive_program_weights, adaptive_program_evidence = receiver_program_weights(
        adaptive_direction,
        adaptive_edges["program_z"].to_numpy(dtype=float),
        program_fit,
    )
    adaptive_program_strict, adaptive_program_calls = _rc4_strict_rankings(
        adaptive_edges,
        strict_template,
        adaptive_program_weights,
        target=target,
        reference=reference,
        method=METHOD_PROGRAM_STRICT,
        semantics=(
            "exact_liana_calls_with_adaptive_multiview_residual_direction_and_"
            "frozen_reliability_shrunk_receiver_program_weight"
        ),
    )
    if adaptive_program_calls != parity_calls:
        raise AssertionError("adaptive program head changed the LIANA call set")
    adaptive_program_fallback = _rc4_coverage_fallback(
        adaptive_program_strict,
        rc2_rankings,
        method=METHOD_PROGRAM_FALLBACK,
        reason_prefix="adaptive_program_not_estimable",
        semantics=(
            "adaptive_program_soft_cardinality_percentile_with_crychic_wald_"
            "percentile_only_when_liana_pair_not_estimable"
        ),
    )

    all_rankings = pd.concat(
        [
            rc2_rankings,
            rc3_strict,
            rc3_fallback,
            adaptive_strict,
            adaptive_fallback,
            adaptive_program_strict,
            adaptive_program_fallback,
        ],
        ignore_index=True,
    )
    scores, coverage, summary = _evaluate(all_rankings, expected, dataset=dataset)
    adaptive_edges["rc3_program_evidence"] = rc3_evidence
    adaptive_edges["rc3_program_weight"] = rc3_weights
    adaptive_edges["adaptive_program_evidence"] = adaptive_program_evidence
    adaptive_edges["adaptive_program_weight"] = adaptive_program_weights
    adaptive_edges["formal_inference_allowed"] = False
    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "directed_edge_adaptive_residuals.parquet": (
            output / "directed_edge_adaptive_residuals.parquet",
            adaptive_edges,
        ),
        "receiver_program_effects.parquet": (
            output / "receiver_program_effects.parquet",
            effects,
        ),
        "view_reliability.tsv": (output / "view_reliability.tsv", view_reliability),
        "cv_diagnostics.tsv": (output / "cv_diagnostics.tsv", cv),
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
        "equal_rc2_parity": parity,
        "adaptive_fit": adaptive_fit.to_dict(),
        "program_fit": program_fit.to_dict(),
        "adaptive_sign_flips_vs_rc2": int(
            adaptive_edges["adaptive_sign_flipped_vs_rc2"].sum()
        ),
        "selected_calls": adaptive_calls,
        "backbone_modified": False,
        "liana_call_set_modified": False,
        "benchmark_head_modified": True,
        "formal_release_allowed": False,
        "limitations": [
            "adaptive view reliability is edge-CV predictive, not formal inference",
            "receiver-program source is partial-pipeline cross-fit evidence",
            "all new outputs are benchmark rankings with formal inference disabled",
        ],
        "inputs": {
            "rc2": rc2_provenance,
            "program": program_provenance,
            "config": _input_record(config_path),
            "acceptance": _input_record(acceptance_path),
            "rc3_frozen_candidate": _input_record(rc3_frozen_path),
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
    parser.add_argument("--rc2-run", type=Path, required=True)
    parser.add_argument("--program-run", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--rc3-frozen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        rc2_run=args.rc2_run,
        program_run=args.program_run,
        expected_path=args.expected,
        config_path=args.config,
        acceptance_path=args.acceptance,
        rc3_frozen_path=args.rc3_frozen,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "adaptive_fit": result["adaptive_fit"],
                "adaptive_sign_flips_vs_rc2": result["adaptive_sign_flips_vs_rc2"],
                "selected_calls": result["selected_calls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
