"""Run the accepted RC2 residual on exact LIANA Kuppe/MS artifacts."""

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
from benchmarks.adapters.liana.run_condition_aware import (
    ALPHA,
    _build_rankings,
)
from benchmarks.adapters.liana.run_condition_aware import (
    SCHEMA_VERSION as LIANA_SCHEMA_VERSION,
)
from benchmarks.literature.component_swap_benchmark import _evaluate
from benchmarks.literature.frozen_signed_cardinality_benchmark import (
    _load_component_run,
)
from benchmarks.literature.liana_hypergraph_residual import (
    conservative_sign_statistic,
    fit_cross_validated_residual,
)

SCHEMA_VERSION = "crychic-liana-hypergraph-residual-real-v1"
METHOD_STRICT = "CRYCHIC_RC2_residual_strict"
METHOD_FALLBACK = "CRYCHIC_RC2_residual_fallback"
METHOD_VERSION = "liana-hypergraph-residual-rc2-v1"
ACCEPTANCE_SHA256 = (
    "6a603eb1ba3a37f67ff48e3ba31fd939aad214d5eb7f913d1e02572183e83b12"
)
CONFIG_SHA256 = "6ecb8c2a5b5381e56f0ac773d36cf7e31663fd1de9096d76c00605586b2ea56d"


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


def _validate_frozen_rc2(config_path: Path, acceptance_path: Path) -> dict[str, Any]:
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("frozen RC2 config checksum mismatch")
    if sha256_file(acceptance_path) != ACCEPTANCE_SHA256:
        raise ValueError("frozen RC2 acceptance checksum mismatch")
    acceptance = _read_json(acceptance_path)
    if acceptance.get("accepted") is not True:
        raise ValueError("RC2 simulation candidate did not pass acceptance")
    return _read_json(config_path)


def _bound_liana_output(
    run: Path, manifest: Mapping[str, Any], filename: str
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(filename), Mapping
    ):
        raise ValueError(f"LIANA run lacks bound output: {filename}")
    record = cast(Mapping[str, Any], outputs[filename])
    path = run / filename
    if record.get("filename") != filename or record.get("sha256") != sha256_file(path):
        raise ValueError(f"LIANA output checksum mismatch: {filename}")
    return path


def _select_calls(
    lr_results: pd.DataFrame,
    sign_statistic: np.ndarray,
    *,
    target: str,
    reference: str,
) -> pd.DataFrame:
    if len(lr_results) != len(sign_statistic):
        raise ValueError("LIANA rows and residual sign statistic length mismatch")
    p_value = pd.to_numeric(
        lr_results["interaction_pvalue"], errors="coerce"
    ).to_numpy(dtype=float)
    selected = (
        np.isfinite(p_value)
        & np.isfinite(sign_statistic)
        & (p_value < ALPHA)
        & (sign_statistic != 0.0)
    )
    result = lr_results.loc[selected].copy()
    result["condition"] = np.where(
        sign_statistic[selected] > 0.0, target, reference
    )
    return result.sort_values(
        ["condition", "source", "target", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def _load_liana_run(
    run: Path,
    *,
    component_manifest: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, object],
]:
    manifest_path = run / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != LIANA_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("LIANA run schema or completion status mismatch")
    analysis_unit = manifest.get("analysis_unit")
    if not isinstance(analysis_unit, Mapping) or (
        analysis_unit.get("replicate_key") != "sample_id"
        or analysis_unit.get("primary_panel") is not True
    ):
        raise ValueError("LIANA run is not the paper-matched sample primary panel")
    lr_path = _bound_liana_output(run, manifest, "liana_lr_results.tsv.gz")
    calls_path = _bound_liana_output(
        run, manifest, "condition_specific_calls.tsv.gz"
    )
    analysis_path = _bound_liana_output(run, manifest, "analysis_cell_types.tsv")
    ranking_path = _bound_liana_output(
        run, manifest, "condition_cell_pair_rankings.tsv"
    )
    component_liana = component_manifest.get("inputs", {}).get(
        "liana_native_ranking"
    )
    if not isinstance(component_liana, Mapping) or component_liana.get(
        "sha256"
    ) != sha256_file(ranking_path):
        raise ValueError("component benchmark is not bound to this LIANA run")
    provenance = {
        "manifest": _input_record(manifest_path),
        "lr_results": _input_record(lr_path),
        "native_calls": _input_record(calls_path),
        "analysis_cell_types": _input_record(analysis_path),
        "native_rankings": _input_record(ranking_path),
    }
    return (
        pd.read_csv(lr_path, sep="\t"),
        pd.read_csv(calls_path, sep="\t"),
        pd.read_csv(analysis_path, sep="\t"),
        manifest,
        provenance,
    )


def _rebuild_rankings(
    calls: pd.DataFrame,
    lr_results: pd.DataFrame,
    analysis: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    prepared = cast(Mapping[str, Any], manifest["prepared_input"])
    contrast = cast(Mapping[str, Any], manifest["contrast"])
    analysis_unit = cast(Mapping[str, Any], manifest["analysis_unit"])
    source_cell_types = tuple(str(value) for value in prepared["source_cell_types"])
    eligible = tuple(
        analysis.loc[analysis["status"].eq("analyzed"), "cell_type"]
        .astype(str)
        .sort_values(kind="stable")
    )
    return _build_rankings(
        calls,
        lr_results,
        source_cell_types=source_cell_types,
        eligible_cell_types=eligible,
        target=str(contrast["target"]),
        reference=str(contrast["reference"]),
        dataset_id=str(manifest["dataset_id"]),
        replicate_key=str(analysis_unit["replicate_key"]),
        subject_key=str(analysis_unit["subject_key"]),
    )


def _assert_native_parity(
    rebuilt_calls: pd.DataFrame,
    native_calls: pd.DataFrame,
    rebuilt_rankings: pd.DataFrame,
    native_rankings: pd.DataFrame,
) -> dict[str, object]:
    call_columns = ["condition", "source", "target", "interaction_id"]
    pd.testing.assert_frame_equal(
        rebuilt_calls.loc[:, call_columns].reset_index(drop=True),
        native_calls.loc[:, call_columns].reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    rebuilt_normalized = rebuilt_rankings.reset_index(drop=True).copy()
    native_normalized = native_rankings.reset_index(drop=True).copy()
    rebuilt_normalized["reason_code"] = rebuilt_normalized[
        "reason_code"
    ].fillna("")
    native_normalized["reason_code"] = native_normalized["reason_code"].fillna("")
    pd.testing.assert_frame_equal(
        rebuilt_normalized,
        native_normalized,
        check_exact=True,
        check_dtype=False,
    )
    return {
        "calls_exact": True,
        "call_rows": len(rebuilt_calls),
        "rankings_exact": True,
        "ranking_rows": len(rebuilt_rankings),
    }


def _align_crychic_anchor(
    lr_results: pd.DataFrame, component_edges: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    effects = component_edges.loc[component_edges["arm"].eq("A")].copy()
    effects["crychic_z"] = np.divide(
        pd.to_numeric(effects["effect"], errors="coerce"),
        pd.to_numeric(effects["standard_error"], errors="coerce"),
        out=np.full(len(effects), np.nan, dtype=float),
        where=pd.to_numeric(effects["standard_error"], errors="coerce") > 0.0,
    )
    keys = ["sender", "receiver", "ligand", "receptor"]
    if effects.duplicated(keys).any():
        raise ValueError("CRYCHIC anchor has duplicate biological edge keys")
    working = lr_results.rename(
        columns={"source": "sender", "target": "receiver"}
    ).copy()
    working = working.merge(
        effects.loc[:, [*keys, "crychic_z"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    working["view_sender"] = working["sender"].astype(str)
    working["view_receiver"] = working["receiver"].astype(str)
    working["view_ligand"] = working["ligand"].astype(str)
    working["view_receptor"] = working["receptor"].astype(str)
    working["view_interaction"] = working["interaction_id"].astype(str)
    finite = working["crychic_z"].notna()
    return working, {
        "liana_edges": len(working),
        "crychic_anchor_edges": int(finite.sum()),
        "crychic_anchor_coverage": float(finite.mean()),
        "baseline_anchor_spearman": working.loc[
            finite, "interaction_stat"
        ].corr(working.loc[finite, "crychic_z"], method="spearman"),
        "baseline_anchor_sign_concordance": float(
            np.sign(working.loc[finite, "interaction_stat"])
            .eq(np.sign(working.loc[finite, "crychic_z"]))
            .mean()
        ),
    }


def _coverage_fallback(
    strict: pd.DataFrame,
    component_rankings: pd.DataFrame,
) -> pd.DataFrame:
    result = strict.copy()
    observed = result["status"].eq("observed")
    result["_strict_rank"] = np.nan
    result.loc[observed, "_strict_rank"] = (
        result.loc[observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = component_rankings.loc[
        component_rankings["method"].eq("C"),
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
    fallback = fallback.rename(
        columns={"estimable_directed_lr": "_fallback_estimable"}
    )
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
            "liana_not_estimable_used_crychic_wald_rank_fallback",
            "liana_and_crychic_fallback_not_estimable",
        ),
    )
    result["method"] = METHOD_FALLBACK
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = (
        "liana_residual_cardinality_percentile_with_crychic_wald_"
        "percentile_only_when_liana_pair_not_estimable"
    )
    return result.drop(
        columns=["_strict_rank", "_fallback_rank", "_fallback_estimable"]
    )


def run(
    *,
    dataset: str,
    liana_run: Path,
    component_run: Path,
    expected_path: Path,
    config_path: Path,
    acceptance_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _validate_frozen_rc2(config_path, acceptance_path)
    output = prepare_output(output_dir, overwrite=overwrite)
    component_edges, component_rankings, component_manifest, component_provenance = (
        _load_component_run(
            component_run.resolve(),
            dataset=dataset,
            expected_path=expected_path.resolve(),
        )
    )
    lr_results, native_calls, analysis, liana_manifest, liana_provenance = (
        _load_liana_run(
            liana_run.resolve(), component_manifest=component_manifest
        )
    )
    contrast = cast(Mapping[str, Any], liana_manifest["contrast"])
    baseline = pd.to_numeric(
        lr_results["interaction_stat"], errors="coerce"
    ).to_numpy(dtype=float)
    rebuilt_calls = _select_calls(
        lr_results,
        baseline,
        target=str(contrast["target"]),
        reference=str(contrast["reference"]),
    )
    rebuilt_rankings = _rebuild_rankings(
        rebuilt_calls, lr_results, analysis, liana_manifest
    )
    native_ranking_path = liana_run / "condition_cell_pair_rankings.tsv"
    parity = _assert_native_parity(
        rebuilt_calls,
        native_calls,
        rebuilt_rankings,
        pd.read_csv(native_ranking_path, sep="\t"),
    )

    aligned, alignment = _align_crychic_anchor(lr_results, component_edges)
    residual_config = cast(Mapping[str, Any], config["residual"])
    theta, fit, cv = fit_cross_validated_residual(
        aligned,
        baseline,
        aligned["crychic_z"].to_numpy(dtype=float),
        views=tuple(str(value) for value in residual_config["views"]),
        key_columns=tuple(
            str(value) for value in residual_config["key_columns"]
        ),
        lambda_grid=tuple(
            float(value) for value in residual_config["lambda_grid"]
        ),
        anchor_grid=tuple(
            float(value) for value in residual_config["anchor_grid"]
        ),
        folds=int(residual_config["folds"]),
        seed=int(residual_config["seed"]),
        minimum_cv_improvement=float(
            residual_config["minimum_cv_improvement"]
        ),
        full_gate_improvement=float(residual_config["full_gate_improvement"]),
    )
    sign_statistic = conservative_sign_statistic(
        baseline,
        theta,
        fit,
        maximum_abs_baseline=float(
            residual_config["sign_correction_max_abs_baseline"]
        ),
    )
    residual_calls = _select_calls(
        lr_results,
        sign_statistic,
        target=str(contrast["target"]),
        reference=str(contrast["reference"]),
    )
    strict = _rebuild_rankings(
        residual_calls, lr_results, analysis, liana_manifest
    )
    strict["method"] = METHOD_STRICT
    strict["method_version"] = METHOD_VERSION
    strict["ranking_semantics"] = (
        "exact_liana_raw_p_lt_0.05_cardinality_with_cv_gated_"
        "multiview_residual_direction"
    )
    fallback = _coverage_fallback(strict, component_rankings)
    all_rankings = pd.concat(
        [component_rankings, strict, fallback], ignore_index=True
    )
    expected = pd.read_csv(expected_path, sep="\t")
    scores, coverage, summary = _evaluate(
        all_rankings, expected, dataset=dataset
    )
    edge_output = aligned.loc[
        :,
        [
            "sender",
            "receiver",
            "interaction_id",
            "ligand",
            "receptor",
            "interaction_stat",
            "interaction_pvalue",
            "crychic_z",
        ],
    ].copy()
    edge_output["residual_theta"] = theta
    edge_output["final_sign_statistic"] = sign_statistic
    edge_output["baseline_sign"] = np.sign(baseline)
    edge_output["final_sign"] = np.sign(sign_statistic)
    edge_output["sign_flipped"] = edge_output["baseline_sign"].ne(
        edge_output["final_sign"]
    )
    edge_output["probability_status"] = "candidate_unreleased"

    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "directed_edge_residuals.parquet": (
            output / "directed_edge_residuals.parquet",
            edge_output,
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
        "cv_diagnostics.tsv": (output / "cv_diagnostics.tsv", cv),
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
        "native_baseline_parity": parity,
        "alignment": alignment,
        "residual_fit": fit.to_dict(),
        "sign_flips": int(edge_output["sign_flipped"].sum()),
        "backbone_modified": False,
        "benchmark_head_modified": True,
        "formal_release_allowed": False,
        "inputs": {
            "component": component_provenance,
            "liana": liana_provenance,
            "config": _input_record(config_path),
            "acceptance": _input_record(acceptance_path),
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
    parser.add_argument("--liana-run", type=Path, required=True)
    parser.add_argument("--component-run", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        liana_run=args.liana_run,
        component_run=args.component_run,
        expected_path=args.expected,
        config_path=args.config,
        acceptance_path=args.acceptance,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "fit": result["residual_fit"],
                "sign_flips": result["sign_flips"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
